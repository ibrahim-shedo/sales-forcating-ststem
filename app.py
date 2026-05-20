import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Enterprise Sales Forecasting Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CLEAN & PROFESSIONAL CSS
# =========================
st.markdown("""
<style>
    /* Clean white background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Main container */
    .main-container {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Header styling - Clean and modern */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 600;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards - Clean cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #2a5298;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: #1e3c72;
    }
    
    .metric-card .delta {
        font-size: 0.8rem;
        color: #28a745;
    }
    
    /* Insight cards */
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .insight-positive {
        border-left-color: #28a745;
        background: #f8fff9;
    }
    
    .insight-warning {
        border-left-color: #ffc107;
        background: #fffbf0;
    }
    
    .insight-info {
        border-left-color: #17a2b8;
        background: #f0f9ff;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: white;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: #6c757d;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a5298;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
        color: #6c757d;
        font-size: 0.85rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3c72;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a5298;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CACHE DATA LOADING
# =========================
@st.cache_data(ttl=3600)
def load_data(file_path="train.csv"):
    """Load and preprocess data"""
    try:
        import os
        if not os.path.exists(file_path):
            st.warning("⚠️ File not found. Using sample data...")
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            sales = np.random.normal(1000, 200, len(dates)) + \
                    np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 500
            df = pd.DataFrame({
                'Date': dates,
                'Sales': np.abs(sales),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
                'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], len(dates))
            })
            return df
        
        df = pd.read_csv(file_path)
        
        # Find date column
        date_col = None
        exact_date_names = ['Order Date', 'order_date', 'OrderDate', 'Date', 'date']
        for col_name in exact_date_names:
            if col_name in df.columns:
                date_col = col_name
                break
        
        if date_col is None:
            for col in df.columns:
                if 'date' in col.lower() and 'id' not in col.lower():
                    date_col = col
                    break
        
        if date_col is None:
            st.error("❌ No date column found")
            return None
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        if len(df) == 0:
            st.error("❌ No valid dates found")
            return None
        
        df = df.sort_values(date_col)
        
        # Find sales column
        sales_col = None
        exact_sales_names = ['Sales', 'sales', 'Revenue', 'revenue']
        for col_name in exact_sales_names:
            if col_name in df.columns:
                sales_col = col_name
                break
        
        if sales_col is None:
            for col in df.columns:
                if any(kw in col.lower() for kw in ['sales', 'revenue', 'amount']):
                    sales_col = col
                    break
        
        if sales_col is None:
            st.error("❌ No sales column found")
            return None
        
        df.rename(columns={date_col: 'Date', sales_col: 'Sales'}, inplace=True)
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        df = df.dropna(subset=['Sales'])
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# =========================
# METRICS CALCULATION
# =========================
def calculate_advanced_metrics(df):
    """Calculate business metrics"""
    metrics = {}
    
    metrics['Total_Sales'] = df['Sales'].sum()
    metrics['Avg_Sales'] = df['Sales'].mean()
    metrics['Total_Orders'] = len(df)
    metrics['Max_Sales'] = df['Sales'].max()
    metrics['Min_Sales'] = df['Sales'].min()
    metrics['Std_Dev'] = df['Sales'].std()
    metrics['CV'] = (metrics['Std_Dev'] / metrics['Avg_Sales']) * 100 if metrics['Avg_Sales'] > 0 else 0
    
    df['Month'] = df['Date'].dt.month
    
    monthly_sales = df.groupby('Month')['Sales'].sum()
    metrics['Peak_Month'] = monthly_sales.idxmax() if len(monthly_sales) > 0 else None
    metrics['Peak_vs_Low_Ratio'] = monthly_sales.max() / monthly_sales.min() if monthly_sales.min() > 0 else 0
    
    weekly_sales = df.set_index('Date').resample('W')['Sales'].sum()
    if len(weekly_sales) > 4:
        recent_avg = weekly_sales.tail(4).mean()
        prior_avg = weekly_sales.head(4).mean()
        metrics['Trend'] = ((recent_avg - prior_avg) / prior_avg) * 100 if prior_avg > 0 else 0
    
    return metrics

# =========================
# FORECAST MODEL
# =========================
@st.cache_resource(ttl=3600)
def train_forecast_model(df, periods, seasonality_mode='multiplicative'):
    """Train Prophet model"""
    try:
        prophet_df = df.groupby('Date')['Sales'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 30:
            st.warning("⚠️ Need at least 30 days of data")
            return None, None, None
        
        model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.95
        )
        
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, include_history=True)
        forecast = model.predict(future)
        
        return model, forecast, prophet_df
        
    except Exception as e:
        st.error(f"❌ Forecasting error: {str(e)}")
        return None, None, None

# =========================
# MAIN APP
# =========================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Sales Forecasting Suite</h1>
        <p>AI-Powered Predictive Analytics for Business Growth</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Controls")
        st.markdown("---")
        
        # Data source
        data_source = st.radio(
            "Data Source",
            ["Sample Data", "Upload CSV"]
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                df = load_data(tmp_path)
            else:
                st.info("📁 Upload your CSV file")
                df = load_data("train.csv")
        else:
            df = load_data("train.csv")
        
        if df is None:
            st.stop()
        
        # Data info
        st.markdown("---")
        st.markdown("### 📈 Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.date_input(
            "Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        else:
            df_filtered = df.copy()
        
        if 'Region' in df_filtered.columns:
            regions = st.multiselect(
                "Region",
                df_filtered['Region'].unique(),
                default=list(df_filtered['Region'].unique())[:3] if len(df_filtered['Region'].unique()) > 3 else list(df_filtered['Region'].unique())
            )
            if regions:
                df_filtered = df_filtered[df_filtered['Region'].isin(regions)]
        
        if 'Category' in df_filtered.columns:
            categories = st.multiselect(
                "Category",
                df_filtered['Category'].unique(),
                default=list(df_filtered['Category'].unique())[:3] if len(df_filtered['Category'].unique()) > 3 else list(df_filtered['Category'].unique())
            )
            if categories:
                df_filtered = df_filtered[df_filtered['Category'].isin(categories)]
        
        st.markdown("---")
        
        # Forecast settings
        st.markdown("### 🎯 Forecast Settings")
        periods = st.slider("Forecast Horizon (Days)", 7, 365, 90)
        seasonality = st.selectbox("Seasonality", ['multiplicative', 'additive'])
        
        forecast_button = st.button("🚀 Generate Forecast", use_container_width=True)
    
    # Main content
    if df_filtered.empty:
        st.warning("⚠️ No data available. Please adjust filters.")
        st.stop()
    
    # KPI Dashboard
    st.markdown("## 📊 Key Metrics")
    metrics = calculate_advanced_metrics(df_filtered)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Total Sales</h3>
            <div class="value">${metrics['Total_Sales']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Average Sale</h3>
            <div class="value">${metrics['Avg_Sales']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦 Total Orders</h3>
            <div class="value">{metrics['Total_Orders']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        trend_icon = "📈" if metrics.get('Trend', 0) > 0 else "📉"
        trend_color = "#28a745" if metrics.get('Trend', 0) > 0 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{trend_icon} Growth Trend</h3>
            <div class="value" style="color: {trend_color};">{metrics.get('Trend', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Explorer", 
        "🔮 Forecast", 
        "💡 Insights",
        "📥 Export"
    ])
    
    with tab1:
        st.markdown("### Historical Sales Trend")
        
        daily_sales = df_filtered.groupby('Date')['Sales'].sum().reset_index()
        
        fig = px.line(
            daily_sales, 
            x='Date', 
            y='Sales',
            title="Daily Sales Over Time",
            template="simple_white"
        )
        fig.update_traces(line=dict(color='#2a5298', width=2))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_sales = df_filtered.groupby(df_filtered['Date'].dt.month)['Sales'].sum()
            fig_monthly = px.bar(
                x=monthly_sales.index, 
                y=monthly_sales.values,
                title="Monthly Sales Pattern",
                labels={'x': 'Month', 'y': 'Total Sales ($)'},
                template="simple_white"
            )
            fig_monthly.update_traces(marker_color='#2a5298')
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            df_filtered['Weekday'] = df_filtered['Date'].dt.dayofweek
            weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly_sales = df_filtered.groupby('Weekday')['Sales'].sum()
            fig_weekly = px.bar(
                x=weekday_names, 
                y=weekly_sales.values,
                title="Weekly Sales Pattern",
                labels={'x': 'Day', 'y': 'Total Sales ($)'},
                template="simple_white"
            )
            fig_weekly.update_traces(marker_color='#1e3c72')
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        with st.expander("View Raw Data"):
            st.dataframe(df_filtered.head(100), use_container_width=True)
    
    with tab2:
        st.markdown("### Sales Forecast")
        
        if forecast_button:
            with st.spinner("Training forecast model..."):
                model, forecast, prophet_df = train_forecast_model(df_filtered, periods, seasonality)
            
            if model and forecast is not None:
                fig_forecast = go.Figure()
                
                fig_forecast.add_trace(go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#2a5298', width=2)
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#dc3545', width=2, dash='dash')
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(220, 53, 69, 0.2)',
                    line=dict(color='rgba(220, 53, 69, 0)'),
                    name='95% CI'
                ))
                
                fig_forecast.update_layout(
                    title=f"Forecast - Next {periods} Days",
                    xaxis_title="Date",
                    yaxis_title="Sales ($)",
                    template="simple_white",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast summary
                col1, col2, col3 = st.columns(3)
                avg_forecast = forecast['yhat'].tail(periods).mean()
                max_forecast = forecast['yhat'].tail(periods).max()
                min_forecast = forecast['yhat'].tail(periods).min()
                
                with col1:
                    st.metric("Average Forecast", f"${avg_forecast:,.2f}")
                with col2:
                    st.metric("Peak Forecast", f"${max_forecast:,.2f}")
                with col3:
                    st.metric("Low Forecast", f"${min_forecast:,.2f}")
                
                # Store in session state
                st.session_state['forecast'] = forecast
                st.session_state['forecast_periods'] = periods
            else:
                st.error("Could not generate forecast")
        else:
            st.info("👈 Click 'Generate Forecast' to start")
    
    with tab3:
        st.markdown("### Business Insights")
        
        if forecast_button and 'forecast' in st.session_state:
            forecast = st.session_state['forecast']
            periods = st.session_state['forecast_periods']
            
            avg_historical = df_filtered['Sales'].mean()
            avg_forecast = forecast['yhat'].tail(periods).mean()
            growth_rate = ((avg_forecast - avg_historical) / avg_historical) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                if growth_rate > 0:
                    st.markdown(f"""
                    <div class="insight-card insight-positive">
                        <h3>📈 Growth Opportunity</h3>
                        <p>Sales projected to grow by <strong>{growth_rate:.1f}%</strong></p>
                        <p>✅ Recommendation: Increase inventory and marketing spend</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="insight-card insight-warning">
                        <h3>📉 Trend Alert</h3>
                        <p>Sales may decline by <strong>{abs(growth_rate):.1f}%</strong></p>
                        <p>⚠️ Recommendation: Consider promotions and review strategy</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if metrics['CV'] > 50:
                    st.markdown(f"""
                    <div class="insight-card insight-warning">
                        <h3>⚠️ High Volatility</h3>
                        <p>CV: <strong>{metrics['CV']:.0f}%</strong> indicates unstable sales</p>
                        <p>✅ Recommendation: Build safety stock buffers</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="insight-card insight-positive">
                        <h3>✅ Stable Operations</h3>
                        <p>CV: <strong>{metrics['CV']:.0f}%</strong> - Low volatility</p>
                        <p>✅ Maintain current strategy</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Seasonal insights
            if metrics['Peak_vs_Low_Ratio'] > 2:
                st.markdown(f"""
                <div class="insight-card insight-info">
                    <h3>🌟 Peak Season Opportunity</h3>
                    <p>Peak months are <strong>{metrics['Peak_vs_Low_Ratio']:.1f}x</strong> higher than low months</p>
                    <p>✅ Action: Increase inventory by 200% before Month {metrics['Peak_Month']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Anomaly detection
            daily_sales = df_filtered.groupby('Date')['Sales'].sum().reset_index()
            rolling_mean = daily_sales['Sales'].rolling(window=30, center=True).mean()
            rolling_std = daily_sales['Sales'].rolling(window=30, center=True).std()
            anomalies = daily_sales[
                (daily_sales['Sales'] > rolling_mean + 2*rolling_std) |
                (daily_sales['Sales'] < rolling_mean - 2*rolling_std)
            ].dropna()
            
            if len(anomalies) > 0:
                st.warning(f"⚠️ Detected {len(anomalies)} unusual patterns in your data")
                st.dataframe(anomalies, use_container_width=True)
        else:
            st.info("👈 Generate a forecast to see insights")
    
    with tab4:
        st.markdown("### Export Data")
        
        export_type = st.selectbox(
            "Select Data to Export",
            ["Forecast Data", "Historical Data", "Summary Metrics"]
        )
        
        if export_type == "Forecast Data" and 'forecast' in st.session_state:
            export_df = st.session_state['forecast'].tail(st.session_state['forecast_periods'])[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            export_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "forecast.csv", "text/csv")
        elif export_type == "Historical Data":
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "historical_data.csv", "text/csv")
        elif export_type == "Summary Metrics":
            metrics_df = pd.DataFrame([metrics]).T.reset_index()
            metrics_df.columns = ['Metric', 'Value']
            csv = metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "metrics.csv", "text/csv")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by Prophet | Real-time Analytics | AI Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
