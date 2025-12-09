# Finance Capstone — Beta, WACC & Crypto Project Analysis  
**Department of Defense – Office of Algorithmic Overreach**  
**Subject:** Why Let Skynet Price Risk When You Can Do It Yourself?

Jacob Clement
Created with assistance from chatGPT
https://chatgpt.com/share/6937a88e-eb04-800a-8033-44a9a2056cc9

This repository contains a collection of Python scripts used for a finance capstone project focused on:

- Pulling market, stock, and risk-free data  
- Estimating CAPM beta (multiple flavors)  
- Building a WACC from market data and user inputs  
- Evaluating a 5-year “stackable crypto” project via NPV & IRR  

Earlier prototypes are preserved for documentation and archaeology purposes (in case future historians need to reconstruct the exact moment humans decided *not* to let Skynet run the capital budgeting process).

---

## 1. Mission Overview

The main operational pipeline now runs through **`full_assignment.py`**, which orchestrates:

1. Data collection from Yahoo Finance & FRED  
2. Beta estimation and CAPM cost of equity  
3. WACC construction using D/E from Yahoo + user-supplied bond yield & tax rate  
4. NPV & IRR analysis for the capstone crypto project  
5. Export of all relevant tables to Excel for documentation and grading

If you’re an overworked MBA student or a nervous DoD analyst trying to prove that Skynet is mispricing risk, **`full_assignment.py` is the button you push.**

---

## 2. Key Scripts (Current Chain of Command)

### 🧠 `full_assignment.py` — Main Orchestrator (Use This)

End-to-end script that:

- Prompts for:
  - Stock ticker (default: `PFE`)
  - Market index (default: `^GSPC`)
  - FRED risk-free series (default: `GS1`)
- Pulls and merges month-end data from:
  - Yahoo Finance (stock + market)
  - FRED (risk-free rate)
- Computes:
  - Simple returns and excess returns
  - Beta via:
    - **Normal OLS** (with intercept)  
    - **Zero-intercept** regression  
    - **Excess-return CAPM regression**
- Calculates **CAPM cost of equity** from the excess-return beta
- Fetches **Debt/Equity (D/E)** from Yahoo Finance and lets the user override it
- Asks the user for:
  - Cost of debt (bond yield — e.g., from TradingView)
  - Corporate tax rate (e.g., derived from the firm’s 10-K)
- Builds the **WACC**:
  - Computes weights  
    - `w_d = D / (D + E)`  
    - `w_e = E / (D + E)`  
  - Shows a **weight check** (`w_d + w_e`) to make sure the world still sums to 1  
  - Applies taxes:  
    - `WACC = w_d * k_d * (1 − T) + w_e * k_e`
- Evaluates the **5-year crypto project**:
  - Builds the cash-flow stream  
  - Computes NPV at WACC  
  - Solves for IRR (via a bisection routine, no external finance libs needed)  
  - Issues an accept/reject decision (in a tone Skynet would find “inefficiently cautious”)
- Writes everything to Excel:
  - `data` sheet (merged returns)  
  - `beta_summary` sheet  
  - `wacc_summary` sheet  
  - `crypto_cash_flows` & `crypto_summary` sheets  

If this script were a T-800, it would be the one holding a calculator instead of a minigun.

---

### 🧮 `full_assignment_2.py` — Alternate Build / Sandbox

- A parallel version used for refactoring, testing, and “what if we break this on purpose?” experimentation.
- May contain slight variations in structure, prompts, or logging.
- Not the primary entry point, but useful if you’re comparing implementations or testing future enhancements.

Think of it as the **T-800 with the safety protocols toggled on**: still dangerous, but only to bugs.

---

### 📊 `date_issue_fix_2.py` — Beta Engine & Date Alignment

This is the core **beta + data-alignment** module that the newer workflow evolved from. It:

- Pulls stock and index price series from Yahoo Finance
- Gets risk-free rates from FRED
- Resamples everything to month-end
- Computes:
  - Monthly returns  
  - Excess returns  
  - Normal, zero-intercept, and excess-return betas  
- Generates regression plots and exports results to Excel

In the Skynet universe, this would be the **core targeting module**: it figures out how sensitive your stock is to the overall market’s mood swings.

---

### 🧮 `date_issue_fix.py` — Legacy Fix

- Earlier fixed version of the date-alignment and return-calculation pipeline.
- Kept for comparison, regression testing, and “how did we get here?” post-mortems.
- The logic has been superseded by `date_issue_fix_2.py` and `full_assignment.py`, but the file remains as a historical artifact.

---

## 3. Supporting & Exploratory Scripts

### `beta_lab_streamlit.py` — Future Front-End (WIP)

- Prototype Streamlit app to wrap the beta/WACC engine in a simple UI.
- Intended features:
  - User-selectable tickers, date ranges, and RF series
  - Interactive beta plots
  - Display of CAPM equations and regression stats
- Current status: **Not fully operational** — like a Skynet prototype still stuck in UAT.

---

### `calculate_returns.py`

- Early playground for:
  - Date parsing
  - Daily vs. monthly returns
  - Intermediate cleaning and transformations
- Useful if you want to see how the return logic was tested before being baked into the main engine.

---

### `CAPM.py`

- Exploratory implementation of:
  - CAPM formulas  
  - Covariance/variance mechanics  
  - Simple beta calculations
- Good for reviewing the theory and math that later got integrated into `date_issue_fix_2.py` and `full_assignment.py`.

---

### `pull_t_bills.py`

- Prototype script for fetching Treasury bill yields from FRED.
- Superseded by the consolidated FRED handling in the newer scripts.
- Still handy if you want a standalone RF puller.

---

### `pull_cigna_price.py` & `pull_pfizer.py`

- Early scripts for fetching individual stock prices (Cigna, Pfizer).
- Useful examples of how to use `yfinance` for single-ticker pulls.
- Kept for reference and for the inevitable “We need to rerun that one weird test from October” scenario.

---

## 4. Typical Workflow (a.k.a. “How To Not Summon Skynet”)

1. **Clone repo & install dependencies**  
   Make sure you have:
   - Python 3.x  
   - `pandas`, `numpy`, `yfinance`, `fredapi`, `statsmodels`, `openpyxl`, `matplotlib`, `python-dateutil`

2. **Set your FRED API key**  
   In `full_assignment.py`:
   ```python
   FRED_API_KEY = "YOUR_REAL_KEY_HERE"
