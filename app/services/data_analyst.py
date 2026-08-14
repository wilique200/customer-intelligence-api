"""
Data Analyst module logic — generic auto-EDA, forecasting, and
Gemini-backed chat. Ported as-is from main.py. Unlike ml_models.py,
there's no fixed trained model here: this works on whatever CSV is
uploaded, which is why there's no "load model" step in this file.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from app.services import gemini_client


def detect_column_types(df: pd.DataFrame) -> dict:
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
            continue
        try:
            pd.to_datetime(df[col], errors="raise")
            types[col] = "datetime"
        except Exception:
            types[col] = "categorical"
    return types


def auto_eda(df: pd.DataFrame):
    types = detect_column_types(df)
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": [],
    }
    for col, dtype in types.items():
        col_info = {"name": col, "type": dtype, "missing_pct": round(float(df[col].isna().mean()) * 100, 1)}
        if dtype == "numeric":
            series = df[col].dropna()
            if len(series):
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
                col_info.update({
                    "mean": round(float(series.mean()), 2), "median": round(float(series.median()), 2),
                    "min": round(float(series.min()), 2), "max": round(float(series.max()), 2),
                    "outlier_count": outliers,
                })
        elif dtype == "categorical":
            mode = df[col].mode()
            col_info.update({
                "unique_count": int(df[col].nunique()),
                "top_value": str(mode.iloc[0]) if not mode.empty else None,
            })
        summary["columns"].append(col_info)

    missing_penalty = float(df.isna().mean().mean()) * 50
    dup_penalty = (summary["duplicate_rows"] / max(1, len(df))) * 50
    summary["data_quality_score"] = round(max(0, 100 - missing_penalty - dup_penalty), 1)

    summary["correlations"] = compute_correlations(df, types)
    summary["category_breakdowns"] = compute_category_breakdowns(df, types)

    return summary, types


def compute_correlations(df: pd.DataFrame, types: dict, top_n: int = 10):
    """Top pairwise correlations among numeric columns — capped at top_n
    so the payload stays small on wide datasets."""
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    if len(numeric_cols) < 2:
        return []

    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    seen = set()
    for col_a in numeric_cols:
        for col_b in numeric_cols:
            if col_a == col_b or (col_b, col_a) in seen:
                continue
            seen.add((col_a, col_b))
            value = corr_matrix.loc[col_a, col_b]
            if pd.notna(value):
                pairs.append({"column_a": col_a, "column_b": col_b, "correlation": round(float(value), 3)})

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs[:top_n]


def compute_category_breakdowns(df: pd.DataFrame, types: dict, max_categories: int = 3, max_cardinality: int = 15):
    """Mean of each numeric column grouped by category, for a handful of
    categorical columns with reasonable cardinality — capped on both
    axes so this can't blow up on a wide, high-cardinality dataset."""
    categorical_cols = [c for c, t in types.items() if t == "categorical" and df[c].nunique() <= max_cardinality]
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    if not categorical_cols or not numeric_cols:
        return []

    breakdowns = []
    for cat_col in categorical_cols[:max_categories]:
        grouped = df.groupby(cat_col)[numeric_cols].mean(numeric_only=True)
        rows = []
        for group_value, row in grouped.iterrows():
            rows.append({
                "group": str(group_value),
                "count": int((df[cat_col] == group_value).sum()),
                **{col: round(float(val), 2) for col, val in row.items() if pd.notna(val)},
            })
        breakdowns.append({"category_column": cat_col, "groups": rows})
    return breakdowns


def attempt_forecast(df: pd.DataFrame, types: dict):
    datetime_cols = [c for c, t in types.items() if t == "datetime"]
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    if not datetime_cols or not numeric_cols:
        return None

    date_col, value_col = datetime_cols[0], numeric_cols[0]
    ts = df[[date_col, value_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.sort_values(date_col).set_index(date_col)[value_col].resample("ME").sum()

    if len(ts) < 4:
        return None

    try:
        model = ExponentialSmoothing(ts, trend="add", seasonal=None).fit()
        forecast = model.forecast(3)
        return {
            "date_column": date_col,
            "value_column": value_col,
            "history": [{"date": str(d.date()), "value": round(float(v), 2)} for d, v in ts.items()],
            "forecast": [{"date": str(d.date()), "value": round(float(v), 2)} for d, v in forecast.items()],
        }
    except Exception as e:
        print(f"WARNING: forecast failed, omitting it: {e}")
        return None


def analyze_dataset(df: pd.DataFrame):
    summary, types = auto_eda(df)
    forecast = attempt_forecast(df, types)
    return summary, forecast


def train_quick_model(df: pd.DataFrame, target_column: str):
    """Fast, classical model on a user-chosen target — not deep learning,
    deliberately. Same evidence from the churn MLP experiment applies
    here: gradient boosting matches or beats a neural net on tabular
    data, and it trains in seconds inside a live request instead of
    needing a background job queue this app doesn't have.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
    from sklearn.preprocessing import LabelEncoder

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in the dataset.")

    work = df.dropna(subset=[target_column]).copy()
    if len(work) < 20:
        raise ValueError("Not enough non-missing rows in the target column to train a reliable model (need at least 20).")

    y = work[target_column]
    X = work.drop(columns=[target_column])

    # Task type: numeric with many distinct values -> regression;
    # otherwise -> classification (covers numeric categoricals too, e.g.
    # a 1-5 rating column, which "many distinct values" would misjudge).
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 15
    target_encoder = None
    if not is_regression:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    # Simple, safe feature prep: encode categoricals, impute numeric NaNs
    # with the median. Good enough for a quick model, not a full pipeline.
    encoders = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).fillna("missing")
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col])
            encoders[col] = enc

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if is_regression:
        model = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric_name, metric_value = "R²", round(float(r2_score(y_test, preds)), 3)
        secondary_name, secondary_value = "Mean absolute error", round(float(mean_absolute_error(y_test, preds)), 2)
    else:
        model = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric_name, metric_value = "Accuracy", round(float(accuracy_score(y_test, preds)) * 100, 1)
        secondary_name, secondary_value = "Classes predicted", int(len(set(preds)))

    importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda p: p[1], reverse=True
    )
    top_features = [{"feature": f, "importance": int(imp)} for f, imp in importances[:8] if imp > 0]

    return {
        "task_type": "regression" if is_regression else "classification",
        "target_column": target_column,
        "rows_used": len(work),
        "primary_metric": {"name": metric_name, "value": metric_value},
        "secondary_metric": {"name": secondary_name, "value": secondary_value},
        "top_features": top_features,
    }


def ask_gemini_about_data(question: str, summary: dict, forecast: dict) -> str:
    return gemini_client.answer_data_question(question, summary, forecast)
