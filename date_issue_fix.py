## Created with assistance from ChatGPT
## Jacob Clement
## https://chatgpt.com/share/693611a6-ada8-800a-8a2a-3127027e6601

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta


# -----------------------------------------------------------
# CONFIG: put your FRED API key here
# -----------------------------------------------------------
FRED_API_KEY = "290571b3503560d831cb298c225e6cb1"   # <-- CHANGE THIS


# -----------------------------------------------------------
# DATA PULL HELPERS
# -----------------------------------------------------------
def get_month_end_prices_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily data from Yahoo Finance and resample to month-end.
    Uses adjusted prices via auto_adjust=True, then grabs the Close column.
    Works whether yfinance returns a normal Index or a MultiIndex.
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,   # prices already adjusted
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker or date range.")

    # If columns are MultiIndex, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join([str(c) for c in col if c]) for col in data.columns.values]

    # Look for a Close-like column
    close_candidates = [c for c in data.columns if "Close" in c]
    if not close_candidates:
        raise ValueError(f"No Close/Adj Close column found for {ticker}. Got columns: {data.columns}")

    close_col = close_candidates[0]

    # Month-end prices ('ME' = month end; 'M' is deprecated)
    monthly = data[close_col].resample("ME").last()

    # Return as a DataFrame with the ticker as column name
    monthly = monthly.to_frame(name=ticker)
    
    return monthly


def get_month_end_fred(series_id: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Download FRED series and resample to month-end.
    Assumes the series is in annualized percent terms (e.g., 5.00 = 5%).
    """
    fred = Fred(api_key=api_key)

    # FRED series IDs are usually uppercase (e.g., TB3MS, GS1)
    series_id_clean = series_id.upper()

    series = fred.get_series(series_id_clean, observation_start=start, observation_end=end)

    if series is None or len(series) == 0:
        raise ValueError(f"No FRED data returned for {series_id_clean}. Check the series ID or date range.")

    df = series.to_frame(name=series_id_clean)
    df.index = pd.to_datetime(df.index)

    # Month-end values
    monthly = df.resample("ME").last()
    return monthly


# -----------------------------------------------------------
# BETA CALCULATION
# -----------------------------------------------------------
def compute_betas(df: pd.DataFrame, stock_col: str, mkt_col: str, rf_col: str):
    """
    Given a dataframe with stock price, market price, and RF level,
    compute:
      - simple returns
      - excess returns
      - normal regression beta (with intercept)
      - zero-intercept beta
      - excess-return beta (CAPM-style)
    Returns a dict of results and the enriched dataframe.
    """
    # Simple returns
    df["stock_ret"] = df[stock_col].pct_change()
    df["mkt_ret"] = df[mkt_col].pct_change()

    # Monthly RF (assumes annual percent, e.g. 5.00 = 5%)
    df["rf_monthly"] = df[rf_col] / 100.0 / 12.0

    # Drop first row with NaNs
    df = df.dropna(subset=["stock_ret", "mkt_ret", "rf_monthly"]).copy()

    # Excess returns
    df["excess_stock"] = df["stock_ret"] - df["rf_monthly"]
    df["excess_mkt"] = df["mkt_ret"] - df["rf_monthly"]

    # ---------- Normal regression beta (simple returns) ----------
    X = sm.add_constant(df["mkt_ret"])
    y = df["stock_ret"]
    reg_normal = sm.OLS(y, X).fit()
    alpha_normal = reg_normal.params["const"]
    beta_normal = reg_normal.params["mkt_ret"]

    # ---------- Zero-intercept beta (simple returns) ----------
    X0 = df[["mkt_ret"]]  # no constant
    reg_zero = sm.OLS(y, X0).fit()
    beta_zero = reg_zero.params["mkt_ret"]

    # ---------- Excess-return beta (CAPM-style) ----------
    X_ex = sm.add_constant(df["excess_mkt"])
    y_ex = df["excess_stock"]
    reg_excess = sm.OLS(y_ex, X_ex).fit()
    alpha_excess = reg_excess.params["const"]
    beta_excess = reg_excess.params["excess_mkt"]

    results = {
        "data": df,
        "reg_normal": reg_normal,
        "reg_zero": reg_zero,
        "reg_excess": reg_excess,
        "alpha_normal": alpha_normal,
        "beta_normal": beta_normal,
        "beta_zero": beta_zero,
        "alpha_excess": alpha_excess,
        "beta_excess": beta_excess,
    }
    return results


# -----------------------------------------------------------
# PLOTTING HELPERS
# -----------------------------------------------------------
def plot_regression(df, x_col, y_col, alpha, beta, title):
    """
    Scatter plot of returns with regression line.
    """
    x = df[x_col]
    y = df[y_col]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6, label="Monthly observations")

    # Regression line
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = alpha + beta * x_vals
    plt.plot(x_vals, y_vals, label=f"Fit: y = {alpha:.4f} + {beta:.4f} x")

    plt.xlabel("Market Return")
    plt.ylabel("Stock Return")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_regression_zero(df, x_col, y_col, beta, title):
    """
    Scatter plot of returns with zero-intercept regression line.
    """
    x = df[x_col]
    y = df[y_col]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6, label="Monthly observations")

    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = beta * x_vals
    plt.plot(x_vals, y_vals, label=f"Fit: y = {beta:.4f} x")

    plt.xlabel("Market Return")
    plt.ylabel("Stock Return")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# MAIN DRIVER
# -----------------------------------------------------------
def run_beta_engine(
    stock_ticker="PFE",
    market_ticker="^GSPC",
    rf_series="TB3MS",
    years_back=5,
):
    # Date range: last `years_back` years to today
    end_date = datetime.today().date()
    start_date = (datetime.today() - relativedelta(years=years_back)).date()

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    print(f"Using date range: {start_str} to {end_str}")
    print(f"Stock: {stock_ticker}, Market: {market_ticker}, RF: {rf_series}")

    # Pull data
    stock_prices = get_month_end_prices_yahoo(stock_ticker, start_str, end_str)
    mkt_prices = get_month_end_prices_yahoo(market_ticker, start_str, end_str)
    rf_data = get_month_end_fred(rf_series, start_str, end_str, FRED_API_KEY)

    # Merge on month-end index
    df = stock_prices.join(mkt_prices, how="inner").join(rf_data, how="inner")

    # Compute betas
    rf_col_name = rf_data.columns[0]   # e.g. "TB3MS" or "GS1"
    results = compute_betas(df, stock_ticker, market_ticker, rf_col_name)

    # Unpack
    beta_normal = results["beta_normal"]
    alpha_normal = results["alpha_normal"]
    beta_zero = results["beta_zero"]
    beta_excess = results["beta_excess"]
    alpha_excess = results["alpha_excess"]
    data = results["data"]

    # Print equations
    print("\n==================== BETA RESULTS ====================")
    print(f"Normal regression (simple returns):")
    print(f"  R_i = {alpha_normal:.6f} + {beta_normal:.4f} * R_m")
    print(f"  Beta_normal = {beta_normal:.4f}")

    print("\nZero-intercept regression (simple returns):")
    print(f"  R_i = {beta_zero:.4f} * R_m")
    print(f"  Beta_zero = {beta_zero:.4f}")

    print("\nExcess-return regression (CAPM-style):")
    print(f"  (R_i - R_f) = {alpha_excess:.6f} + {beta_excess:.4f} * (R_m - R_f)")
    print(f"  Beta_excess = {beta_excess:.4f}")
    print("======================================================\n")

    # Optional: CAPM expected return using excess beta
    rf_mean = data["rf_monthly"].mean()
    mkt_mean = data["mkt_ret"].mean()
    capm_expected = rf_mean + beta_excess * (mkt_mean - rf_mean)

    print(f"Average monthly RF: {rf_mean:.4%}")
    print(f"Average monthly market return: {mkt_mean:.4%}")
    print(f"CAPM expected monthly return (using beta_excess): {capm_expected:.4%}")

    # Plots
    plot_regression(
        data,
        x_col="mkt_ret",
        y_col="stock_ret",
        alpha=alpha_normal,
        beta=beta_normal,
        title=f"{stock_ticker} vs {market_ticker} (Normal Regression)",
    )

    plot_regression_zero(
        data,
        x_col="mkt_ret",
        y_col="stock_ret",
        beta=beta_zero,
        title=f"{stock_ticker} vs {market_ticker} (Zero-Intercept Regression)",
    )


if __name__ == "__main__":
    # Prompt version
    stock = input("Enter stock ticker (default PFE): ").strip() or "PFE"
    market = input("Enter market index (default ^GSPC): ").strip() or "^GSPC"
    rf = input("Enter FRED RF series (default GS1): ").strip() or "GS1"

    run_beta_engine(stock_ticker=stock, market_ticker=market, rf_series=rf)
