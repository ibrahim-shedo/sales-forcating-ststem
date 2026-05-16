import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# =========================
# APP CONFIG
# =========================

st.set_page_config(page_title="Sales Forecasting System", layout="wide")

st.title("📈 Sales Forecasting & Business Intelligence Dashboard")
st.write("End-to-end forecasting system with KPIs, filters, and insights")

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    # clean columns
    df.columns = df.columns.str.strip()

    # detect date column
    date_col = None
    for col in df.columns:
        if "order" in col.lower() and "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        st.error("No Order Date column found")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df = df.rename(columns={date_col: "Order Date"})

    return df

df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("🎛 Filters")

if "Region" in df.columns:
    region = st.sidebar.selectbox("Region", df["Region"].unique())
else:
    region = None

if "Category" in df.columns:
    category = st.sidebar.selectbox("Category", df["Category"].unique())
else:
    category = None

df_filtered = df.copy()

if region:
    df_filtered = df_filtered[df_filtered["Region"] == region]

if category:
    df_filtered = df_filtered[df_filtered["Category"] == category]

# =========================
# KPI DASHBOARD
# =========================

st.subheader("📊 Key Performance Indicators")

total_sales = df_filtered["Sales"].sum()
avg_sales = df_filtered["Sales"].mean()
orders = len(df_filtered)

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{total_sales:,.0f}")
col2.metric("Average Sales", f"{avg_sales:,.2f}")
col3.metric("Total Orders", orders)

# =========================
# PEAK MONTH
# =========================

monthly = df_filtered.groupby(df_filtered["Order Date"].dt.to_period("M"))["Sales"].sum()
peak_month = monthly.idxmax()

st.success(f"🔥 Peak Sales Month: {peak_month}")

# =========================
# PREPARE DATA FOR PROPHET
# =========================

sales = df_filtered.groupby("Order Date")["Sales"].sum().reset_index()
sales.columns = ["ds", "y"]

st.subheader("📊 Data Preview")
st.dataframe(df_filtered.head())

st.subheader("📈 Aggregated Sales")
st.dataframe(sales.head())

# =========================
# FORECAST SETTINGS
# =========================

periods = st.slider("Forecast Horizon (Days)", 7, 180, 30)

# =========================
# MODEL
# =========================

model = Prophet()
model.fit(sales)

future = model.make_future_dataframe(periods=periods)
forecast = model.predict(future)

# =========================
# FORECAST OUTPUT
# =========================

st.subheader("📊 Forecast Results")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("📈 Trend & Seasonality")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# =========================
# BUSINESS INSIGHT
# =========================

avg_forecast = forecast.tail(periods)["yhat"].mean()
growth_rate = ((avg_forecast - sales["y"].mean()) / sales["y"].mean()) * 100

st.subheader("💡 Business Insights")

st.write(f"📊 Forecasted Average Sales: {avg_forecast:.2f}")
st.write(f"📈 Growth Rate: {growth_rate:.2f}%")

if avg_forecast > sales["y"].mean():
    st.success("📈 Sales expected to grow → increase stock & marketing")
else:
    st.warning("📉 Sales may decline → consider promotions")

# =========================
# EXPORT FEATURE
# =========================

st.subheader("📥 Export Forecast")

csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Forecast CSV",
    csv,
    "sales_forecast.csv",
    "text/csv"
)