import os
from datetime import datetime

import clickhouse_connect
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="Bitcoin Real-time Big Data",
    page_icon="",
    layout="wide",
)


def env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@st.cache_resource
def get_client():
    return clickhouse_connect.get_client(
        host=env("CH_HOST", "localhost"),
        port=int(env("CH_PORT", "8125")),
        username=env("CLICKHOUSE_USER", "default"),
        password=env("CLICKHOUSE_PASSWORD"),
        database=env("CLICKHOUSE_DB", "bigdata"),
    )


@st.cache_data(ttl=10)
def query_df(sql: str) -> pd.DataFrame:
    return get_client().query_df(sql)


def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
    return df


def line_chart(df: pd.DataFrame, y_cols: list[str], height: int = 320) -> None:
    df = normalize_time(df)
    if df.empty:
        st.info("No data for this panel.")
        return
    plot_df = df.set_index("time")[y_cols]
    st.line_chart(plot_df, height=height)


def bar_chart(df: pd.DataFrame, y_col: str, height: int = 320) -> None:
    df = normalize_time(df)
    if df.empty:
        st.info("No data for this panel.")
        return
    st.bar_chart(df.set_index("time")[[y_col]], height=height)


def metric_row() -> None:
    sql = """
    SELECT
      (SELECT count() FROM bigdata.bitcoin_orders WHERE time >= now() - INTERVAL 7 DAY) AS realtime_orders_7d,
      (SELECT count() FROM bigdata.bitcoin_realtime_predictions WHERE time >= now() - INTERVAL 7 DAY) AS realtime_predictions_7d,
      (SELECT count() FROM bigdata.bitcoin_orders_1h_agg) AS hourly_buckets,
      (SELECT max(time) FROM bigdata.bitcoin_orders) AS latest_order_time
    """
    df = query_df(sql)
    if df.empty:
        return
    row = df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realtime Orders 7d", f"{int(row['realtime_orders_7d']):,}")
    c2.metric("Realtime Predictions 7d", f"{int(row['realtime_predictions_7d']):,}")
    c3.metric("Hourly Buckets", f"{int(row['hourly_buckets']):,}")
    latest = row["latest_order_time"]
    if isinstance(latest, pd.Timestamp):
        latest = latest.strftime("%Y-%m-%d %H:%M:%S")
    c4.metric("Latest Order", str(latest))


RAW_CLOSE_SQL = """
SELECT
  time,
  close
FROM bigdata.bitcoin_orders
WHERE time >= now() - INTERVAL 7 DAY
ORDER BY time
"""

RAW_VOLUME_SQL = """
SELECT
  time,
  volume
FROM bigdata.bitcoin_orders
WHERE time >= now() - INTERVAL 7 DAY
ORDER BY time
"""

RAW_ACTUAL_PRED_SQL = """
SELECT
  actual.time AS time,
  actual.actual_close AS actual_close,
  predicted.predicted_close AS predicted_close
FROM
(
  SELECT
    time,
    close AS actual_close
  FROM bigdata.bitcoin_orders
  WHERE time >= now() - INTERVAL 7 DAY
) AS actual
LEFT JOIN
(
  SELECT
    time,
    close_prediction AS predicted_close
  FROM bigdata.bitcoin_realtime_predictions
  WHERE time >= now() - INTERVAL 7 DAY
) AS predicted
ON actual.time = predicted.time
ORDER BY time
"""

RAW_ERROR_SQL = """
SELECT
  actual.time AS time,
  abs(actual.actual_close - predicted.predicted_close) AS abs_error
FROM
(
  SELECT
    time,
    close AS actual_close
  FROM bigdata.bitcoin_orders
  WHERE time >= now() - INTERVAL 7 DAY
) AS actual
INNER JOIN
(
  SELECT
    time,
    close_prediction AS predicted_close
  FROM bigdata.bitcoin_realtime_predictions
  WHERE time >= now() - INTERVAL 7 DAY
) AS predicted
ON actual.time = predicted.time
ORDER BY time
"""

RAW_FEATURES_SQL = """
SELECT
  time,
  close_delta,
  close_std_60,
  dist_to_mean_60,
  volume_sum_60
FROM bigdata.bitcoin_features
WHERE time >= now() - INTERVAL 7 DAY
ORDER BY time
"""

HOURLY_CLOSE_SQL = """
SELECT
  time,
  avgMerge(avg_close_state) AS close
FROM bigdata.bitcoin_orders_1h_agg
GROUP BY time
ORDER BY time
"""

HOURLY_VOLUME_SQL = """
SELECT
  time,
  sumMerge(volume_state) AS volume
FROM bigdata.bitcoin_orders_1h_agg
GROUP BY time
ORDER BY time
"""

HOURLY_ACTUAL_PRED_SQL = """
SELECT
  actual.time AS time,
  actual.actual_close AS actual_close,
  nullIf(predicted.predicted_close, 0) AS predicted_close
FROM
(
  SELECT
    time,
    avgMerge(avg_close_state) AS actual_close
  FROM bigdata.bitcoin_orders_1h_agg
  GROUP BY time
) AS actual
LEFT JOIN
(
  SELECT
    time,
    avgMerge(close_prediction_state) AS predicted_close
  FROM bigdata.bitcoin_predictions_1h_agg
  GROUP BY time
) AS predicted
ON actual.time = predicted.time
ORDER BY time
"""

HOURLY_ERROR_SQL = """
SELECT
  actual.time AS time,
  abs(actual.actual_close - predicted.predicted_close) AS abs_error
FROM
(
  SELECT
    time,
    avgMerge(avg_close_state) AS actual_close
  FROM bigdata.bitcoin_orders_1h_agg
  GROUP BY time
) AS actual
INNER JOIN
(
  SELECT
    time,
    avgMerge(close_prediction_state) AS predicted_close
  FROM bigdata.bitcoin_predictions_1h_agg
  GROUP BY time
) AS predicted
ON actual.time = predicted.time
ORDER BY time
"""

HOURLY_FEATURES_SQL = """
SELECT
  time,
  avgMerge(close_delta_state) AS close_delta,
  avgMerge(close_std_60_state) AS close_std_60,
  avgMerge(dist_to_mean_60_state) AS dist_to_mean_60,
  avgMerge(volume_sum_60_state) AS volume_sum_60
FROM bigdata.bitcoin_features_1h_agg
GROUP BY time
ORDER BY time
"""


st.title("Bitcoin Real-time Big Data Dashboard")
st.caption(f"Rendered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    metric_row()

    realtime_tab, hourly_tab = st.tabs(["Realtime", "Hourly Historical"])

    with realtime_tab:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Bitcoin Close Price")
            line_chart(query_df(RAW_CLOSE_SQL), ["close"])
        with right:
            st.subheader("Trading Volume")
            bar_chart(query_df(RAW_VOLUME_SQL), "volume")

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Actual vs Realtime Predicted Close")
            line_chart(query_df(RAW_ACTUAL_PRED_SQL), ["actual_close", "predicted_close"])
        with right:
            st.subheader("Realtime Prediction Error")
            line_chart(query_df(RAW_ERROR_SQL), ["abs_error"])

        st.subheader("Feature Monitoring")
        line_chart(
            query_df(RAW_FEATURES_SQL),
            ["close_delta", "close_std_60", "dist_to_mean_60", "volume_sum_60"],
            height=360,
        )

    with hourly_tab:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Bitcoin Close Price")
            line_chart(query_df(HOURLY_CLOSE_SQL), ["close"])
        with right:
            st.subheader("Trading Volume")
            bar_chart(query_df(HOURLY_VOLUME_SQL), "volume")

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Actual vs Predicted Close")
            line_chart(query_df(HOURLY_ACTUAL_PRED_SQL), ["actual_close", "predicted_close"])
        with right:
            st.subheader("Prediction Error")
            line_chart(query_df(HOURLY_ERROR_SQL), ["abs_error"])

        st.subheader("Feature Monitoring")
        line_chart(
            query_df(HOURLY_FEATURES_SQL),
            ["close_delta", "close_std_60", "dist_to_mean_60", "volume_sum_60"],
            height=360,
        )

except Exception as exc:
    st.error("Failed to load dashboard data.")
    st.exception(exc)
