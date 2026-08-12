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
    return summary, types


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


def ask_gemini_about_data(question: str, summary: dict, forecast: dict) -> str:
    return gemini_client.answer_data_question(question, summary, forecast)
