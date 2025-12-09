## Created with assistance from ChatGPT
## Jacob Clement
## Updated version with WACC + project NPV/IRR helper
## https://chatgpt.com/share/6937a88e-eb04-800a-8033-44a9a2056cc9


import pandas as pd
import numpy as np

import yfinance as yf
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os  # for saving output files
from typing import Dict, List, Optional

# -----------------------------------------------------------
# CONFIG: put your FRED API key here
# -----------------------------------------------------------
FRED_API_KEY = "Your API Key"   # <-- CHANGE THIS


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
def compute_betas(df: pd.DataFrame, stock_col: str, mkt_col: str, rf_col: str) -> Dict[str, object]:
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
def plot_regression(df: pd.DataFrame, x_col: str, y_col: str, alpha: float, beta: float, title: str) -> None:
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


def plot_regression_zero(df: pd.DataFrame, x_col: str, y_col: str, beta: float, title: str) -> None:
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
# SIMPLE IRR + CRYPTO PROJECT EVAL
# -----------------------------------------------------------
def _npv(rate: float, cash_flows: List[float]) -> float:
    """
    Standard NPV given a discount rate and list of cash flows.
    CF_0 is at t=0, CF_1 at t=1, etc.
    """
    return sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cash_flows))


def _irr(
    cash_flows: List[float],
    guess_low: float = -0.9999,
    guess_high: float = 10.0,
    tol: float = 1e-7,
    max_iter: int = 10_000,
) -> Optional[float]:
    """
    Very simple bisection IRR solver so we don't depend on numpy_financial.
    Returns the rate r such that NPV(r) ~= 0, or None if it fails.
    """
    # Ensure the cash flows change sign (otherwise IRR not defined in the usual way)
    if all(cf >= 0 for cf in cash_flows) or all(cf <= 0 for cf in cash_flows):
        return None

    low, high = guess_low, guess_high
    npv_low = _npv(low, cash_flows)
    npv_high = _npv(high, cash_flows)

    # If the signs are not opposite, IRR may be outside [low, high]
    if npv_low * npv_high > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        npv_mid = _npv(mid, cash_flows)

        if abs(npv_mid) < tol:
            return mid

        # Bisection step
        if npv_low * npv_mid < 0:
            high, npv_high = mid, npv_mid
        else:
            low, npv_low = mid, npv_mid

    # Did not converge
    return None


def evaluate_crypto_project(
    wacc: float,
    initial_investment: float = 1_000_000.0,
    annual_interest_rate: float = 0.35,
    years: int = 5,
    terminal_value: float = 1_200_000.0,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Evaluate the "stackable crypto" project from the assignment.

    Assumptions (default values match the project description):
        - Initial investment of $1,000,000 at t = 0.
        - 35% *simple* interest paid out each year on the original principal.
        - Horizon of 5 years.
        - At the end of year 5, you also receive a terminal value of $1,200,000
          (e.g., liquidation value of the crypto position).
        - Discount rate is the firm's WACC.

    Cash flow pattern (with defaults):
        t = 0: -1,000,000
        t = 1..4: +350,000
        t = 5: +350,000 + 1,200,000
    """
    # Build cash-flow list
    coupon = initial_investment * annual_interest_rate
    cash_flows = [-initial_investment]
    for t in range(1, years):
        cash_flows.append(coupon)
    # Final year: interest + terminal value
    cash_flows.append(coupon + terminal_value)

    npv_value = _npv(wacc, cash_flows)
    irr_value = _irr(cash_flows)

    # Small table for Excel / printing
    periods = list(range(0, years + 1))
    cf_table = pd.DataFrame({"t": periods, "cash_flow": cash_flows})

    result = {
        "wacc": wacc,
        "cash_flows": cash_flows,
        "npv": npv_value,
        "irr": irr_value,
        "cash_flow_table": cf_table,
    }

    # Optionally append results to the Excel file created by run_beta_engine
    if output_path is not None:
        try:
            with pd.ExcelWriter(
                output_path,
                mode="a",
                engine="openpyxl",
                if_sheet_exists="replace",
            ) as writer:
                cf_table.to_excel(writer, sheet_name="crypto_cash_flows", index=False)
                summary_df = pd.DataFrame(
                    {
                        "WACC": [wacc],
                        "NPV": [npv_value],
                        "IRR": [irr_value],
                        "initial_investment": [initial_investment],
                        "annual_interest_rate": [annual_interest_rate],
                        "years": [years],
                        "terminal_value": [terminal_value],
                    }
                )
                summary_df.to_excel(writer, sheet_name="crypto_summary", index=False)
        except Exception as e:
            print(f"Warning: could not write project sheets to Excel: {e}")

    return result


# -----------------------------------------------------------
# MAIN DRIVER
# -----------------------------------------------------------
def run_beta_engine(
    stock_ticker: str = "PFE",
    market_ticker: str = "^GSPC",
    rf_series: str = "GS1",
    years_back: int = 5,
) -> str:
    """
    Pulls data, computes betas, CAPM cost of equity, and writes out an Excel file.
    Returns the path to the Excel file so we can reuse it when we add project results.
    """
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
    print("Normal regression (simple returns):")
    print(f"  R_i = {alpha_normal:.6f} + {beta_normal:.4f} * R_m")
    print(f"  Beta_normal = {beta_normal:.4f}")

    print("\nZero-intercept regression (simple returns):")
    print(f"  R_i = {beta_zero:.4f} * R_m")
    print(f"  Beta_zero = {beta_zero:.4f}")

    print("\nExcess-return regression (CAPM-style):")
    print(f"  (R_i - R_f) = {alpha_excess:.6f} + {beta_excess:.4f} * (R_m - R_f)")
    print(f"  Beta_excess = {beta_excess:.4f}")
    print("======================================================\n")

    # -----------------------------------------------------------
    # CAPM COST OF EQUITY (based on excess-return beta)
    # -----------------------------------------------------------
    rf_mean = data["rf_monthly"].mean()
    mkt_mean = data["mkt_ret"].mean()
    capm_expected = rf_mean + beta_excess * (mkt_mean - rf_mean)

    print(f"Average monthly RF: {rf_mean:.4%}")
    print(f"Average monthly market return: {mkt_mean:.4%}")
    print(f"CAPM expected monthly return (using beta_excess): {capm_expected:.4%}")

    # -----------------------------------------------------------
    # SAVE OUTPUT DATA TO EXCEL
    # -----------------------------------------------------------

    # Make sure data folder exists
    os.makedirs("data", exist_ok=True)

    # Annualize the monthly averages
    rf_mean_annual = rf_mean * 12
    mkt_mean_annual = mkt_mean * 12

    def capm_cost_of_capital(beta: float) -> float:
        return rf_mean_annual + beta * (mkt_mean_annual - rf_mean_annual)

    summary = pd.DataFrame(
        {
            "alpha_normal": [alpha_normal],
            "beta_normal": [beta_normal],
            "beta_zero": [beta_zero],
            "alpha_excess": [alpha_excess],
            "beta_excess": [beta_excess],
            "avg_monthly_rf": [rf_mean],
            "avg_monthly_market_ret": [mkt_mean],
            "avg_annual_rf": [rf_mean_annual],
            "avg_annual_market_ret": [mkt_mean_annual],
            "cost_of_capital_normal": [capm_cost_of_capital(beta_normal)],
            "cost_of_capital_zero": [capm_cost_of_capital(beta_zero)],
            "cost_of_capital_excess": [capm_cost_of_capital(beta_excess)],
        }
    )

    # Output filename uses tickers + RF series
    # Note: replace characters like '^' in filename
    safe_stock = stock_ticker.replace("^", "")
    safe_mkt = market_ticker.replace("^", "")
    output_path = f"data/{safe_stock}_{safe_mkt}_{rf_series}_beta_data_with_calc_graphs.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="data")
        summary.to_excel(writer, sheet_name="beta_summary", index=False)

    print(f"\nData and beta summary saved to: {output_path}\n")

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

    return output_path


if __name__ == "__main__":
    # -----------------------------
    # 1) Beta + CAPM engine
    # -----------------------------
    stock = input("Enter stock ticker (default PFE): ").strip() or "PFE"
    market = input("Enter market index (default ^GSPC): ").strip() or "^GSPC"
    rf = input("Enter FRED RF series (default GS1): ").strip() or "GS1"

    excel_path = run_beta_engine(stock_ticker=stock, market_ticker=market, rf_series=rf)

    # -----------------------------
    # 2) Crypto project evaluation
    # -----------------------------
    print("\nNow let's evaluate the 5-year crypto project from the assignment.")
    print("Press ENTER to accept the default shown in brackets.\n")

    # NOTE: Default WACC below (0.0722) is just an example based on your Excel work.
    # Feel free to override it each time you run the script.
    wacc_str = input("Enter WACC as a decimal [0.0722]: ").strip()
    wacc = float(wacc_str) if wacc_str else 0.0722

    init_str = input("Initial investment [1000000]: ").strip()
    initial_investment = float(init_str) if init_str else 1_000_000.0

    r_str = input("Annual interest rate (e.g., 0.35 for 35%) [0.35]: ").strip()
    annual_rate = float(r_str) if r_str else 0.35

    years_str = input("Number of years [5]: ").strip()
    years = int(years_str) if years_str else 5

    term_str = input("Terminal value at final year [1200000]: ").strip()
    terminal_value = float(term_str) if term_str else 1_200_000.0

    proj_results = evaluate_crypto_project(
        wacc=wacc,
        initial_investment=initial_investment,
        annual_interest_rate=annual_rate,
        years=years,
        terminal_value=terminal_value,
        output_path=excel_path,
    )

    print("\n================= CRYPTO PROJECT RESULTS =================")
    print(f"WACC (discount rate): {proj_results['wacc']:.4%}")
    print(f"Cash flows: {proj_results['cash_flows']}")
    print(f"NPV at WACC: ${proj_results['npv']:,.2f}")

    irr_val = proj_results["irr"]
    if irr_val is not None:
        print(f"IRR: {irr_val:.4%}")
    else:
        print("IRR: not defined (cash flows do not change sign in a standard way).")

    if proj_results["npv"] > 0 and irr_val is not None and irr_val > wacc:
        decision = "ACCEPT the project (NPV > 0 and IRR > WACC)."
    elif proj_results["npv"] > 0:
        decision = "ACCEPT the project (NPV > 0)."
    else:
        decision = "REJECT the project (NPV < 0)."

    print(f"Decision rule: {decision}")
    print("Crypto project results have been added to the Excel file (crypto_* sheets) if possible.")
