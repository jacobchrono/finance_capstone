# beta_lab_app.py
# work in progress Streamlit app for CAPM beta calculation lab
#
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from fredapi import Fred

# -------------------------------------------------------------------
# RISK-FREE SERIES (FRED IDs)
# -------------------------------------------------------------------
RISK_FREE_SERIES = {
    "3M T-Bill (DTB3, daily secondary market)": "DTB3",
    "6M T-Bill (DTB6, daily secondary market)": "DTB6",
    "1Y Treasury (DGS1, constant maturity)": "DGS1",
    "3Y Treasury (DGS3, constant maturity)": "DGS3",
}

# -------------------------------------------------------------------
# HELPERS TO LOAD DATA
# -------------------------------------------------------------------
@st.cache_data
def load_price_data(ticker: str, start, end, freq: str) -> pd.DataFrame:
    """
    Download price data from Yahoo Finance for a given ticker.

    Args:
        ticker: e.g. 'CI'
        start, end: datetime.date
        freq: 'Daily' or 'Monthly'

    Returns:
        DataFrame with single column 'Close', indexed by Date.
    """
    interval = "1d" if freq == "Daily" else "1mo"

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No price data returned for {ticker}. Check ticker or dates.")

    data = data.dropna()
    df = data[["Close"]].copy()
    df.columns = [ticker]
    return df


@st.cache_data
def load_risk_free_from_fred(
    series_id: str,
    start,
    end,
    freq: str,
    fred_api_key: str,
) -> pd.Series:
    """
    Load risk-free yield from FRED, return as a Series indexed by Date.
    The series is an ANNUAL yield in PERCENT.

    For Monthly frequency, resample to month-end average.
    """
    if not fred_api_key:
        # Try environment variable as fallback
        fred_api_key = os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise ValueError(
            "No FRED API key provided. Set FRED_API_KEY env var or enter it in the sidebar."
        )

    fred = Fred(api_key=fred_api_key)

    rf = fred.get_series(
        series_id,
        observation_start=start,
        observation_end=end,
    )

    if rf is None or len(rf) == 0:
        raise ValueError(f"No FRED data returned for series {series_id}.")

    rf = rf.to_frame(name="yield")
    rf.index = pd.to_datetime(rf.index)

    # If we're working monthly, aggregate daily yields to monthly
    if freq == "Monthly":
        rf = rf.resample("M").mean()

    rf = rf.sort_index()
    return rf["yield"]  # percent per annum


def compute_excess_returns(
    ticker: str,
    market_ticker: str,
    start,
    end,
    freq: str,
    rf_series_id: str,
    fred_api_key: str,
) -> pd.DataFrame:
    """
    Pull stock + market prices, compute returns, pull risk-free from FRED,
    and return a DataFrame with excess returns.

    Columns:
        R_i, R_m, R_f, Excess_i, Excess_m
    """
    # 1) Prices
    stock_prices = load_price_data(ticker, start, end, freq)
    market_prices = load_price_data(market_ticker, start, end, freq)

    # 2) Simple returns
    stock_ret = stock_prices[ticker].pct_change()
    market_ret = market_prices[market_ticker].pct_change()

    returns = pd.concat(
        [stock_ret.rename("R_i"), market_ret.rename("R_m")],
        axis=1,
        join="inner",
    ).dropna()

    if returns.empty:
        raise ValueError("No overlapping return data between stock and market.")

    # 3) Risk-free from FRED (annual yield in percent)
    rf_yields_annual_pct = load_risk_free_from_fred(
        rf_series_id, start, end, freq, fred_api_key
    )

    # Align & fill
    rf_yields_annual_pct = rf_yields_annual_pct.reindex(returns.index).ffill().bfill()

    # Convert percent p.a. -> decimal p.a.
    rf_yields_annual = rf_yields_annual_pct / 100.0

    # Convert to per-period rate
    periods_per_year = 252 if freq == "Daily" else 12
    rf_per_period = rf_yields_annual / periods_per_year

    returns["R_f"] = rf_per_period

    # 4) Excess returns
    returns["Excess_i"] = returns["R_i"] - returns["R_f"]
    returns["Excess_m"] = returns["R_m"] - returns["R_f"]

    returns = returns.dropna()

    if returns.empty:
        raise ValueError("No valid excess-return observations after alignment.")

    return returns


# -------------------------------------------------------------------
# BETA CALCULATION
# -------------------------------------------------------------------
def compute_betas(excess_df: pd.DataFrame):
    """
    Given columns 'Excess_i' and 'Excess_m', compute:

    1) OLS with intercept:
         (R_i - R_f)_t = alpha + beta * (R_M - R_f)_t + eps
       where:
         beta = Cov(x, y) / Var(x)
         alpha = mean(y) - beta * mean(x)

    2) OLS forced through origin:
         (R_i - R_f)_t = beta_0 * (R_M - R_f)_t + eps
       where:
         beta_0 = sum(x*y) / sum(x^2)
    """
    x = excess_df["Excess_m"]
    y = excess_df["Excess_i"]

    # With intercept
    cov_xy = y.cov(x)
    var_x = x.var()
    beta_with_intercept = cov_xy / var_x
    alpha_with_intercept = y.mean() - beta_with_intercept * x.mean()

    # Through origin
    beta_through_origin = (x * y).sum() / (x**2).sum()

    return alpha_with_intercept, beta_with_intercept, beta_through_origin


def make_beta_plot(
    excess_df: pd.DataFrame,
    alpha: float,
    beta: float,
    beta0: float,
    ticker: str,
    market_ticker: str,
    freq: str,
):
    """
    Scatter of Excess_i vs Excess_m with two regression lines:
      - with intercept (alpha + beta * x)
      - through origin (beta0 * x)
    """
    x = excess_df["Excess_m"]
    y = excess_df["Excess_i"]

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.6, label="Observed periods")

    x_line = np.linspace(x.min(), x.max(), 200)

    # Line with intercept
    y_line_intercept = alpha + beta * x_line
    ax.plot(
        x_line,
        y_line_intercept,
        linewidth=2,
        label="Regression (with intercept)",
    )

    # Line through origin
    y_line_origin = beta0 * x_line
    ax.plot(
        x_line,
        y_line_origin,
        linewidth=2,
        linestyle="--",
        label="Regression (through origin, α=0)",
    )

    ax.set_xlabel(r"Market excess return $(R_M - R_f)$")
    ax.set_ylabel(rf"{ticker} excess return $(R_i - R_f)$")
    ax.set_title(
        f"Beta estimation for {ticker} vs {market_ticker} ({freq.lower()} data)"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig


# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
def main():
    st.title("CAPM Beta Lab 🧪")
    st.write(
        """
        This app:
        - Pulls **stock & market prices** from Yahoo Finance  
        - Pulls **T-bill / Treasury yields** from FRED as your risk-free rate  
        - Computes **excess returns** for both stock and market  
        - Estimates **two betas**:
            1. OLS with intercept  
            2. OLS with intercept forced to zero  
        - Plots both **Security Characteristic Lines** and shows the **equations**
        """
    )

    # -------------------------
    # Sidebar Inputs
    # -------------------------
    st.sidebar.header("Inputs")

    default_start = date(2015, 1, 1)
    default_end = date.today()

    ticker = st.sidebar.text_input("Stock ticker", value="CI")

    market_label = st.sidebar.selectbox(
        "Market index",
        [
            "S&P 500 (^GSPC)",
            "Dow Jones (^DJI)",
            "Nasdaq (^IXIC)",
        ],
        index=0,
    )
    market_map = {
        "S&P 500 (^GSPC)": "^GSPC",
        "Dow Jones (^DJI)": "^DJI",
        "Nasdaq (^IXIC)": "^IXIC",
    }
    market_ticker = market_map[market_label]

    freq = st.sidebar.radio("Data frequency", ["Daily", "Monthly"], index=0)
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=default_end)

    # Risk-free via FRED
    rf_label = st.sidebar.selectbox(
        "Risk-free (FRED series)",
        list(RISK_FREE_SERIES.keys()),
        index=0,
    )
    rf_series_id = RISK_FREE_SERIES[rf_label]

    fred_key_input = st.sidebar.text_input(
        "FRED API key (leave blank to use FRED_API_KEY env var)",
        type="password",
        value="",
    )

    st.sidebar.write("---")
    run_button = st.sidebar.button("Run beta calculation")

    # -------------------------
    # MAIN AREA
    # -------------------------
    if not run_button:
        st.info("Configure inputs in the sidebar and click **Run beta calculation**.")
        return

    try:
        excess_df = compute_excess_returns(
            ticker=ticker.upper(),
            market_ticker=market_ticker,
            start=start_date,
            end=end_date,
            freq=freq,
            rf_series_id=rf_series_id,
            fred_api_key=fred_key_input,
        )

        alpha, beta_standard, beta_origin = compute_betas(excess_df)

    except Exception as e:
        st.error(f"Error computing beta: {e}")
        return

    st.subheader("Results")

    st.write(f"**Ticker:** {ticker.upper()}")
    st.write(f"**Market index:** {market_ticker} ({market_label})")
    st.write(f"**Frequency:** {freq}")
    st.write(f"**Risk-free series (FRED):** `{rf_series_id}` – {rf_label}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Beta (OLS with intercept)", f"{beta_standard:.4f}")
    with col2:
        st.metric("Alpha (OLS with intercept)", f"{alpha:.4f}")
    with col3:
        st.metric("Beta (through origin, α=0)", f"{beta_origin:.4f}")

    st.write("---")

    # -------------------------
    # Equations
    # -------------------------
    st.markdown("### CAPM & Beta Equations")

    # Core CAPM
    st.latex(
        r"\mathbb{E}[R_i] = R_f + \beta\left(\mathbb{E}[R_M] - R_f\right)"
    )

    # OLS with intercept
    st.latex(
        r"(R_i - R_f)_t = \alpha + \beta (R_M - R_f)_t + \varepsilon_t"
    )

    # Through-origin regression
    st.latex(
        r"(R_i - R_f)_t = \beta_0 (R_M - R_f)_t + \varepsilon_t,\quad \alpha = 0"
    )

    st.markdown("#### Your estimated equations")

    st.latex(
        rf"(R_i - R_f)_t = {alpha:.4f} + {beta_standard:.4f}\,(R_M - R_f)_t + \varepsilon_t"
    )

    st.latex(
        rf"(R_i - R_f)_t = {beta_origin:.4f}\,(R_M - R_f)_t + \varepsilon_t \quad (\alpha = 0)"
    )

    st.write("---")

    # -------------------------
    # Plot
    # -------------------------
    st.markdown("### Security Characteristic Lines (SCL)")

    fig = make_beta_plot(
        excess_df=excess_df,
        alpha=alpha,
        beta=beta_standard,
        beta0=beta_origin,
        ticker=ticker.upper(),
        market_ticker=market_ticker,
        freq=freq,
    )
    st.pyplot(fig)

    # -------------------------
    # Data peek
    # -------------------------
    st.markdown("### Sample of excess-return data")
    st.dataframe(
        excess_df[["R_i", "R_m", "R_f", "Excess_i", "Excess_m"]].head(15)
    )


if __name__ == "__main__":
    main()
