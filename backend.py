from typing import List
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import yfinance as yf

app = FastAPI(
    title="CAPM Analytics Engine API",
    description="REST API for systematic risk estimation and OLS linear regression",
    version="1.0.0",
)


# --- Pydantic Data Models (Request/Response Contracts) ---
class CAPMRequest(BaseModel):
  ticker: str = Field(..., example="RELIANCE.NS")
  benchmark: str = Field(default="^NSEI", example="^NSEI")
  period: str = Field(default="3y", example="3y")
  annual_rf_pct: float = Field(default=4.5, ge=0.0, le=20.0, example=4.5)
  rolling_window: int = Field(default=60, ge=10, le=252, example=60)


class ScatterPoint(BaseModel):
  x: float  # Benchmark excess return
  y: float  # Stock excess return


class RollingPoint(BaseModel):
  date: str
  beta: float


class CAPMResponse(BaseModel):
  ticker: str
  benchmark: str
  beta: float
  alpha_annual_pct: float
  r_squared_pct: float
  alpha_p_value: float
  beta_p_value: float
  durbin_watson: float
  kurtosis: float
  scatter_data: List[ScatterPoint]
  rolling_beta_data: List[RollingPoint]


# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
  return {"status": "healthy", "service": "CAPM Analytics Engine"}


# --- Core Analytics Endpoint ---
@app.post("/api/capm", response_model=CAPMResponse)
def calculate_capm(payload: CAPMRequest):
  ticker = payload.ticker.upper().strip()
  benchmark = payload.benchmark.upper().strip()

  # 1. Fetch market data
  try:
    df_raw = yf.download(
        [ticker, benchmark], period=payload.period, auto_adjust=True
    )["Close"]
    df_prices = df_raw.dropna()
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Failed to fetch market data: {str(e)}"
    )

  if (
      df_prices.empty
      or ticker not in df_prices.columns
      or benchmark not in df_prices.columns
  ):
    raise HTTPException(
        status_code=404,
        detail=f"Tickers '{ticker}' or '{benchmark}' not found or returned no data.",
    )

  # 2. Return Calculations
  df_returns = df_prices.pct_change().dropna()
  rf_daily = (1 + (payload.annual_rf_pct / 100)) ** (1 / 252) - 1

  y_excess = df_returns[ticker] - rf_daily
  x_excess = df_returns[benchmark] - rf_daily

  # 3. OLS Linear Regression
  x_const = sm.add_constant(x_excess)
  model = sm.OLS(y_excess, x_const).fit()

  alpha_daily, beta = float(model.params.iloc[0]), float(model.params.iloc[1])
  alpha_annual_pct = alpha_daily * 252 * 100
  r_squared_pct = float(model.rsquared) * 100
  alpha_p_value = float(model.pvalues.iloc[0])
  beta_p_value = float(model.pvalues.iloc[1])

  # Statistical Diagnostics
  residuals = model.resid
  dw_stat = float(durbin_watson(residuals))
  kurt_stat = float(residuals.kurtosis()) + 3.0  # Excess to absolute kurtosis

  # 4. Rolling Beta Calculation
  cov_series = (
      df_returns[ticker]
      .rolling(window=payload.rolling_window)
      .cov(df_returns[benchmark])
  )
  var_series = df_returns[benchmark].rolling(window=payload.rolling_window).var()
  rolling_beta_series = (cov_series / var_series).dropna()

  # 5. Prepare Output Payloads
  scatter_data = [
      ScatterPoint(x=float(x_val), y=float(y_val))
      for x_val, y_val in zip(x_excess, y_excess)
  ]

  rolling_beta_data = [
      RollingPoint(date=str(date.date()), beta=float(b_val))
      for date, b_val in rolling_beta_series.items()
  ]

  return CAPMResponse(
      ticker=ticker,
      benchmark=benchmark,
      beta=round(beta, 3),
      alpha_annual_pct=round(alpha_annual_pct, 2),
      r_squared_pct=round(r_squared_pct, 2),
      alpha_p_value=round(alpha_p_value, 4),
      beta_p_value=round(beta_p_value, 4),
      durbin_watson=round(dw_stat, 3),
      kurtosis=round(kurt_stat, 3),
      scatter_data=scatter_data,
      rolling_beta_data=rolling_beta_data,
  )