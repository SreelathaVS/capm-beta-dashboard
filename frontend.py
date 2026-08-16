import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="CAPM Microservice Dashboard", page_icon="⚡", layout="wide"
)

API_URL = "http://127.0.0.1:8000/api/capm"

st.title("⚡ Quantitative CAPM Dashboard")
st.caption(
    "Decoupled Microservice: Streamlit UI connected to FastAPI backend via REST."
)

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.header("Model Inputs")
ticker = st.sidebar.text_input("Stock Ticker", value="RELIANCE.NS").upper().strip()
benchmark = (
    st.sidebar.text_input("Benchmark Index", value="^NSEI").upper().strip()
)
period = st.sidebar.selectbox(
    "Lookback Period", ["1y", "2y", "3y", "5y"], index=2
)
rolling_window = st.sidebar.slider(
    "Rolling Window (Days)", min_value=20, max_value=120, value=60, step=10
)
rf_annual_pct = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.5
)

# ----------------- CALL FASTAPI BACKEND -----------------
payload = {
    "ticker": ticker,
    "benchmark": benchmark,
    "period": period,
    "annual_rf_pct": rf_annual_pct,
    "rolling_window": rolling_window,
}

try:
  with st.spinner("Calling FastAPI analytics microservice..."):
    response = requests.post(API_URL, json=payload, timeout=15)

  if response.status_code == 200:
    data = response.json()

    # 1. Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beta (Systematic Risk)", f"{data['beta']:.2f}")
    c2.metric("Annualized Alpha", f"{data['alpha_annual_pct']:.2f}%")
    c3.metric("R-Squared", f"{data['r_squared_pct']:.2f}%")
    c4.metric(
        "Alpha p-Value",
        f"{data['alpha_p_value']:.4f}",
        "Significant" if data["alpha_p_value"] < 0.05 else "Not Significant",
        delta_color="normal" if data["alpha_p_value"] < 0.05 else "off",
    )

    # 2. Tabs for Visuals & Diagnostics
    tab1, tab2, tab3 = st.tabs(
        ["Regression Line (SCL)", "Rolling Beta Time Series", "REST API Data"]
    )

    with tab1:
      scatter_df = pd.DataFrame(data["scatter_data"])
      fig_scl = px.scatter(
          scatter_df,
          x="x",
          y="y",
          trendline="ols",
          labels={
              "x": f"{data['benchmark']} Excess Return",
              "y": f"{data['ticker']} Excess Return",
          },
          title=f"Security Characteristic Line: {data['ticker']} vs {data['benchmark']}",
          opacity=0.6,
      )
      st.plotly_chart(fig_scl, use_container_width=True)

    with tab2:
      rolling_df = pd.DataFrame(data["rolling_beta_data"])
      rolling_df["date"] = pd.to_datetime(rolling_df["date"])

      fig_rolling = go.Figure()
      fig_rolling.add_trace(
          go.Scatter(
              x=rolling_df["date"],
              y=rolling_df["beta"],
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
          title=f"Rolling {rolling_window}-Day Beta",
          xaxis_title="Date",
          yaxis_title="Beta",
      )
      st.plotly_chart(fig_rolling, use_container_width=True)

    with tab3:
      st.subheader("Statistical Diagnostics")
      col_diag1, col_diag2 = st.columns(2)
      col_diag1.metric("Durbin-Watson (Autocorrelation)", data["durbin_watson"])
      col_diag2.metric("Kurtosis (Fat Tails)", data["kurtosis"])

      st.subheader("Raw JSON Response from FastAPI")
      st.json({
          k: v
          for k, v in data.items()
          if k not in ["scatter_data", "rolling_beta_data"]
      })

  else:
    st.error(
        f"Backend Error [{response.status_code}]:"
        f" {response.json().get('detail', 'Unknown error')}"
    )

except requests.exceptions.ConnectionError:
  st.error(
      "Could not connect to FastAPI server. Ensure FastAPI is running on"
      " http://127.0.0.1:8000"
  )