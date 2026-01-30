import pandas as pd
import streamlit as st
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "data" / "predictions.csv"


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

st.set_page_config(page_title="NCAAB Predictions", layout="wide")
st.title("NCAAB Daily Predictions")

if not DATA_PATH.exists():
    st.error(f"Missing predictions file: {DATA_PATH}")
    st.stop()

df = load_predictions(DATA_PATH)

if "date" in df.columns and not df["date"].isna().all():
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    selected = st.date_input("Date", value=max_date, min_value=min_date, max_value=max_date)
    view = df[df["date"].dt.date == selected].copy()
else:
    st.warning("No valid 'date' column found; showing all rows.")
    view = df

st.dataframe(view, use_container_width=True)
