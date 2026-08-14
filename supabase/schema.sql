-- Pulse Supabase Schema (v1)
-- Run this in the Supabase SQL editor (Project -> SQL Editor -> New query).
--
-- Design note: Pulse is organized around ORGANIZATIONS rather than
-- individual users directly, since it's aimed at businesses -- a team
-- needs shared access to the same churn/data analyses, not siloed
-- per-person data. Same reasoning FinGuard used for households,
-- applied to a B2B context instead of a family one. Every table is
-- scoped to organization membership via row-level security below.

-- ============================================================
-- CLEANUP FIRST — safe to run whether this is a fresh project or a
-- retry after a partial/failed run. DROP ... IF EXISTS no-ops on a
-- clean project, and clears out any half-built state from before.
-- ============================================================
drop trigger if exists on_auth_user_created on auth.users;
drop function if exists handle_new_user();
drop table if exists expense_records cascade;
drop table if exists data_analyst_chat_messages cascade;
drop table if exists data_analyst_analyses cascade;
drop table if exists churn_analyses cascade;
drop table if exists organization_members cascade;
drop table if exists organizations cascade;
drop function if exists is_org_member(uuid);

-- ============================================================
-- ORGANIZATIONS
-- One per signed-up business, created automatically on signup via the
-- trigger at the bottom of this file. Multiple team members can belong
-- to the same organization (invite-teammates UI is a later phase --
-- the schema already supports it via organization_members).
-- ============================================================
create table organizations (
  id uuid primary key default gen_random_uuid(),
  name text not null default 'My Organization',
  plan text not null default 'free' check (plan in ('free', 'pro', 'enterprise')),
  created_by uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);

create table organization_members (
  organization_id uuid not null references organizations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null default 'owner' check (role in ('owner', 'member')),
  joined_at timestamptz not null default now(),
  primary key (organization_id, user_id)
);

-- ============================================================
-- CHURN ANALYSES
-- One row per uploaded CSV run through the churn model. Per-customer
-- results stored as JSONB rather than a fully normalized child table --
-- simplest thing that works for an MVP (same pragmatic call FinGuard
-- made storing raw_text_excerpt as plain text instead of over-
-- normalizing bill line items).
-- ============================================================
create table churn_analyses (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references organizations(id) on delete cascade,
  uploaded_by uuid not null references auth.users(id) on delete cascade,
  filename text,
  total_customers integer not null,
  flagged_count integer not null,
  threshold numeric(5,2) not null,
  data_completeness_warning text,
  results jsonb not null,  -- array of {customer_id, risk_score, flagged, reason}
  created_at timestamptz not null default now()
);

create index idx_churn_analyses_org_created
  on churn_analyses (organization_id, created_at desc);

-- ============================================================
-- DATA ANALYST ANALYSES
-- One row per uploaded dataset analyzed by the Data Analyst module.
-- ============================================================
create table data_analyst_analyses (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references organizations(id) on delete cascade,
  uploaded_by uuid not null references auth.users(id) on delete cascade,
  filename text,
  summary jsonb not null,
  forecast jsonb,
  created_at timestamptz not null default now()
);

create index idx_data_analyst_analyses_org_created
  on data_analyst_analyses (organization_id, created_at desc);

-- ============================================================
-- DATA ANALYST CHAT MESSAGES
-- Chat history tied to a specific analysis, so a user can leave and
-- come back to the same conversation about their data.
-- ============================================================
create table data_analyst_chat_messages (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null references data_analyst_analyses(id) on delete cascade,
  organization_id uuid not null references organizations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

create index idx_chat_messages_analysis
  on data_analyst_chat_messages (analysis_id, created_at);

-- ============================================================
-- EXPENSE RECORDS
-- One row per receipt/invoice a user confirms after Gemini reads it.
-- ============================================================
create table expense_records (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references organizations(id) on delete cascade,
  uploaded_by uuid not null references auth.users(id) on delete cascade,
  vendor text,
  expense_date date,
  amount numeric(12,2),
  category text,
  currency text not null default 'USD',
  created_at timestamptz not null default now()
);

create index idx_expense_records_org_created
  on expense_records (organization_id, created_at desc);

-- ============================================================
-- ROW LEVEL SECURITY
-- Every table is scoped to organizations the requesting user belongs to.
-- ============================================================
alter table organizations enable row level security;
alter table organization_members enable row level security;
alter table churn_analyses enable row level security;
alter table data_analyst_analyses enable row level security;
alter table data_analyst_chat_messages enable row level security;
alter table expense_records enable row level security;

-- Helper: is the current user a member of this organization?
create or replace function is_org_member(oid uuid)
returns boolean
language sql
security definer
set search_path = public
stable
as $$
  select exists (
    select 1 from public.organization_members
    where organization_id = oid and user_id = auth.uid()
  );
$$;

create policy "members can view their organization"
  on organizations for select
  using (is_org_member(id));

create policy "creator can insert organization"
  on organizations for insert
  with check (created_by = auth.uid());

create policy "members can view membership rows"
  on organization_members for select
  using (is_org_member(organization_id));

create policy "users can add themselves to an org they created"
  on organization_members for insert
  with check (user_id = auth.uid());

create policy "members can view churn analyses"
  on churn_analyses for select
  using (is_org_member(organization_id));

create policy "members can insert churn analyses"
  on churn_analyses for insert
  with check (is_org_member(organization_id));

create policy "members can view data analyst analyses"
  on data_analyst_analyses for select
  using (is_org_member(organization_id));

create policy "members can insert data analyst analyses"
  on data_analyst_analyses for insert
  with check (is_org_member(organization_id));

create policy "members can view chat messages"
  on data_analyst_chat_messages for select
  using (is_org_member(organization_id));

create policy "members can insert chat messages"
  on data_analyst_chat_messages for insert
  with check (is_org_member(organization_id));

create policy "members can view expense records"
  on expense_records for select
  using (is_org_member(organization_id));

create policy "members can insert expense records"
  on expense_records for insert
  with check (is_org_member(organization_id));

-- ============================================================
-- AUTO-CREATE a default organization + membership on signup.
-- This is the single most common place FinGuard's own fix_*.sql files
-- had to patch (fix_signup_trigger.sql) -- worth testing this exact
-- trigger first, by signing up one test user, before testing anything
-- else in this schema.
-- ============================================================
create or replace function handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  new_org_id uuid;
begin
  insert into public.organizations (name, created_by)
  values ('My Organization', new.id)
  returning id into new_org_id;

  insert into public.organization_members (organization_id, user_id, role)
  values (new_org_id, new.id, 'owner');

  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function handle_new_user();
