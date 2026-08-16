import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="CAPM Beta & Alpha Dashboard",
    page_icon="📈",
    layout="wide",
)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Model Inputs")
ticker = st.sidebar.text_input("Stock Ticker", value="RELIANCE.NS").upper().strip()
benchmark = (
    st.sidebar.text_input("Benchmark Index", value="^NSEI").upper().strip()
)
period = st.sidebar.selectbox(
    "Lookback Period", ["1y", "2y", "3y", "5y"], index=2
)
rolling_window = st.sidebar.slider(
    "Rolling Beta Window (Days)",
    min_value=20,
    max_value=120,
    value=60,
    step=10,
)
rf_annual_pct = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.5
)

rf_daily = (1 + (rf_annual_pct / 100)) ** (1 / 252) - 1

st.title("📈 Capital Asset Pricing Model (CAPM) Dashboard")
st.caption(
    "Estimate systematic market risk (Beta), Jensen's excess return (Alpha), and rolling market sensitivity using OLS linear regression."
)


# ----------------- DATA LOADER & CACHING -----------------
@st.cache_data(ttl=3600)
def load_price_data(tickers, period_str):
  data = yf.download(tickers, period=period_str, auto_adjust=True)["Close"]
  return data.dropna()


# ----------------- EXECUTION ENGINE -----------------
try:
  with st.spinner("Fetching market data and running regression..."):
    df_prices = load_price_data([ticker, benchmark], period)

    if (
        df_prices.empty
        or ticker not in df_prices.columns
        or benchmark not in df_prices.columns
    ):
      st.error(
          "Could not retrieve data for the requested tickers. Please verify the"
          " ticker symbols."
      )
      st.stop()

    # Calculate returns and excess returns
    df_returns = df_prices.pct_change().dropna()
    y_excess = df_returns[ticker] - rf_daily
    x_excess = df_returns[benchmark] - rf_daily

    # Fit OLS
    x_const = sm.add_constant(x_excess)
    ols_model = sm.OLS(y_excess, x_const).fit()

    alpha_daily, beta = ols_model.params.iloc[0], ols_model.params.iloc[1]
    alpha_annual = alpha_daily * 252
    r_squared = ols_model.rsquared
    p_val_alpha = ols_model.pvalues.iloc[0]

  # ----------------- SUMMARY METRIC CARDS -----------------
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Beta (Systematic Risk)", f"{beta:.2f}")
  col2.metric("Annualized Alpha", f"{alpha_annual * 100:.2f}%")
  col3.metric("R-Squared (Fit)", f"{r_squared:.2%}")
  col4.metric(
      "Alpha p-Value",
      f"{p_val_alpha:.4f}",
      "Statistically Significant" if p_val_alpha < 0.05 else "Not Significant",
      delta_color="normal" if p_val_alpha < 0.05 else "off",
  )

  # ----------------- CHARTS & VISUALIZATIONS -----------------
  tab1, tab2, tab3 = st.tabs([
      "Security Characteristic Line",
      "Rolling Beta Analysis",
      "OLS Diagnostics",
  ])

  with tab1:
    fig_scl = px.scatter(
        x=x_excess,
        y=y_excess,
        trendline="ols",
        labels={
            "x": f"{benchmark} Daily Excess Return",
            "y": f"{ticker} Daily Excess Return",
        },
        title=f"Security Characteristic Line: {ticker} vs. {benchmark}",
        opacity=0.6,
    )
    fig_scl.update_layout(height=500)
    st.plotly_chart(fig_scl, use_container_width=True)

  with tab2:
    cov_series = (
        df_returns[ticker].rolling(window=rolling_window).cov(df_returns[benchmark])
    )
    var_series = df_returns[benchmark].rolling(window=rolling_window).var()
    rolling_beta = (cov_series / var_series).dropna()

    fig_rolling = go.Figure()
    fig_rolling.add_trace(
        go.Scatter(
            x=rolling_beta.index,
            y=rolling_beta,
            mode="lines",
            name=f"Rolling {rolling_window}D Beta",
        )
    )
    fig_rolling.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Market Beta (1.0)",
    )
    fig_rolling.update_layout(
        title=f"Rolling {rolling_window}-Day Beta Over Time",
        xaxis_title="Date",
        yaxis_title="Beta",
        height=500,
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

  with tab3:
    st.text("Detailed Statistical Output from statsmodels:")
    st.text(ols_model.summary().as_text())

except Exception as e:
  st.error(f"An error occurred during execution: {e}")