# Finance Capstone — Beta, WACC & Crypto Project Analysis  
**Department of Defense – Office of Algorithmic Overreach**  
**Subject:** Why Let Skynet Price Risk When You Can Do It Yourself?

Jacob Clement
Created with assistance from ChatGPT and Claude
https://chatgpt.com/share/6937a88e-eb04-800a-8033-44a9a2056cc9

https://claude.ai/public/artifacts/d46315bc-16c5-4ace-805b-e1bec5952e60


This repository contains a collection of Python scripts used for a finance capstone project focused on:

- Pulling market, stock, and risk-free data  
- Estimating CAPM beta (multiple flavors)  
- Building a WACC from market data and user inputs  
- Evaluating a 5-year "stackable crypto" project via NPV & IRR  

Earlier prototypes are preserved for documentation and archaeology purposes (in case future historians need to reconstruct the exact moment humans decided *not* to let Skynet run the capital budgeting process).

---

## 1. Mission Overview

The main operational pipeline now runs through **`full_assignment.py`**, which orchestrates:

1. Data collection from Yahoo Finance & FRED  
2. Beta estimation and CAPM cost of equity  
3. WACC construction using D/E from Yahoo + user-supplied bond yield & tax rate  
4. NPV & IRR analysis for the capstone crypto project  
5. Export of all relevant tables to Excel for documentation and grading

If you're an overworked MBA student or a nervous DoD analyst trying to prove that Skynet is mispricing risk, **`full_assignment.py` is the button you push.**

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
  - Corporate tax rate (e.g., derived from the firm's 10-K)
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
  - Issues an accept/reject decision (in a tone Skynet would find "inefficiently cautious")
- Writes everything to Excel:
  - `data` sheet (merged returns)  
  - `beta_summary` sheet  
  - `wacc_summary` sheet  
  - `crypto_cash_flows` & `crypto_summary` sheets  

If this script were a T-800, it would be the one holding a calculator instead of a minigun.

---

### 🧮 `full_assignment_2.py` — Alternate Build / Sandbox

- A parallel version used for refactoring, testing, and "what if we break this on purpose?" experimentation.
- May contain slight variations in structure, prompts, or logging.
- Not the primary entry point, but useful if you're comparing implementations or testing future enhancements.

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

In the Skynet universe, this would be the **core targeting module**: it figures out how sensitive your stock is to the overall market's mood swings.

---

### 🧮 `date_issue_fix.py` — Legacy Fix

- Earlier fixed version of the date-alignment and return-calculation pipeline.
- Kept for comparison, regression testing, and "how did we get here?" post-mortems.
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
- Kept for reference and for the inevitable "We need to rerun that one weird test from October" scenario.

---

## 4. Typical Workflow (a.k.a. "How To Not Summon Skynet")

### Step 1: Clone repo & install dependencies

Make sure you have:
- Python 3.x  
- `pandas`, `numpy`, `yfinance`, `fredapi`, `statsmodels`, `openpyxl`, `matplotlib`, `python-dateutil`

```bash
pip install pandas numpy yfinance fredapi statsmodels openpyxl matplotlib python-dateutil
```

### Step 2: Set your FRED API key

In `full_assignment.py`:

```python
FRED_API_KEY = "YOUR_REAL_KEY_HERE"
```

### Step 3: Run the main script

```bash
python full_assignment.py
```

Follow the prompts:
- Accept defaults or enter your own tickers/series
- Confirm D/E (from Yahoo) or override it
- Enter bond yield and tax rate
- Optionally tweak the crypto project inputs

### Step 4: Review the output

Open the generated Excel file in the `data/` folder to:
- Copy tables and charts into your write-up
- Double-check calculations before you submit to a human, not a machine overlord

---

## 5. Notes on Tax Rate & D/E

### D/E Ratio

- Pulled automatically from Yahoo Finance via `yfinance` when possible
- Interpreted carefully (the script checks whether Yahoo is giving you a ratio or a percent-style number)
- You can override it at runtime if you've computed your own D/E from the firm's financials

### Tax Rate

- **Best source:** the firm's latest 10-K / 10-Q
- Compute as `Income tax expense / Income before taxes` for an effective rate
- The script accepts any decimal 0–0.99, so you can use statutory or effective rates as needed

**Remember:** garbage in, garbage out — or as Skynet would say, "Your parameters are suboptimal, human."

---

## 6. Disclaimer from the Office of Algorithmic Overreach

This codebase:

**Does not:**
- Attempt to launch nukes
- Self-modify (beyond your git commits)
- Claim that a 35% crypto yield is realistic in an efficient market

**Does:**
- Pull real financial data
- Produce real betas, WACC, NPV, and IRR
- Give you just enough quantitative firepower to convince your professor that you didn't outsource everything to a time-traveling AI

If anything looks off, assume user error first, model risk second, and Judgment Day third.

---# Finance Capstone — Beta, WACC & Crypto Project Analysis

**Department of Defense – Office of Algorithmic Overreach**  
**Subject:** Why Let Skynet Price Risk When You Can Do It Yourself?

**Jacob Clement**  
Created with assistance from ChatGPT  
https://chatgpt.com/share/6937a88e-eb04-800a-8033-44a9a2056cc9

---

This repository contains a collection of Python scripts used for a finance capstone project focused on:

- Pulling market, stock, and risk-free data
- Estimating CAPM beta (multiple flavors)
- Building a WACC from market data and user inputs
- Evaluating a 5-year "stackable crypto" project via NPV & IRR

Earlier prototypes are preserved for documentation and archaeology purposes (for the future historians who will inevitably try to determine the moment humanity almost let Skynet handle capital budgeting).

---

## 1. Mission Overview

The main operational pipeline now runs through `full_assignment_3.py`, which orchestrates:

- Data collection from Yahoo Finance & FRED
- Beta estimation and CAPM cost of equity
- WACC construction using D/E from Yahoo + user-supplied bond yield & tax rate
- NPV & IRR analysis for the capstone crypto project
- Export of all relevant tables and plots to Excel and PNG for documentation and grading

If you're an overworked MBA student or a DoD analyst trying to prove Skynet is mispricing risk, `full_assignment.py` is the button you push.

---

## 2. Key Scripts (Current Chain of Command)

### 🧠 `full_assignment_3.py` — Main Orchestrator (USE THIS)

This script:

**Prompts for:**
- Stock ticker (default PFE)
- Market index (default ^GSPC)
- FRED risk-free series (default GS1)

**Pulls month-end stock, market, and RF data**

**Computes:**
- Simple monthly returns
- Excess returns
- Four versions of beta:
  - Normal regression (with intercept)
  - Zero-intercept regression
  - Excess-return CAPM regression (with intercept)
  - Excess-return CAPM regression (zero intercept)

**Calculates CAPM cost of equity for each beta flavor**

**Pulls Debt/Equity from Yahoo Finance and allows overrides**

**Prompts for:**
- Bond yield (cost of debt)
- Corporate tax rate

**Builds a full WACC with proper weights**

**Evaluates a 5-year crypto project using:**
- User-provided principal
- Simple annual interest rate (default 5%)
- Terminal value
- Correct discounting using the computed WACC

**Exports:**
- All data tables to Excel
- All graphs to the `graphs/` folder

This is the T-800 of the repo — fully operational, mission-capable, and surprisingly good at corporate finance.

### 🧮 `full_assignment_2.py` & `full_assignment.py` — Experimental Alternates

Parallel versions for sandboxing and testing.  
Think of them as safer T-800 prototypes—same muscles, fewer bad decisions.

### 📊 `date_issue_fix_2.py` — Core Beta Engine

Handles:
- Yahoo + FRED data alignment
- Month-end resampling
- Beta computations
- Regression charts
- Excel output

This is the targeting module that most later scripts evolved out of.

### 🧮 `date_issue_fix.py` — Legacy Version

Kept for historical and debugging purposes.  
Useful for archaeology. Not used in the current workflow.

---

## 3. Supporting & Exploratory Scripts

- `beta_lab_streamlit.py` — UI prototype (WIP)
- `calculate_returns.py` — Early return-testing sandbox
- `CAPM.py` — Manual CAPM & covariance implementation
- `pull_t_bills.py` — Standalone FRED RF tester
- `pull_cigna_price.py` / `pull_pfizer.py` — Single-ticker prototypes

---

## 4. Outputs: Excel & Graphs

### 📁 `final_excel/`

Contains ready-to-submit Excel workbooks, e.g.:

`PFE_GSPC_GS1_beta_data_with_calc_graphs_final.xlsx`

**Sheets include:**
- `data` — merged monthly series
- `beta_summary` — all beta versions
- `wacc_summary`
- `crypto_cash_flows`
- `crypto_summary`

### 📁 `graphs/`

PNG images of all regression plots:
- `PFE_GSPC_GS1_simple_normal.png`
- `PFE_GSPC_GS1_simple_zero.png`
- `PFE_GSPC_GS1_excess_normal.png`
- `PFE_GSPC_GS1_excess_zero.png`

These are the ones to paste directly into the professor-facing write-up.

---

## 5. Typical Workflow (a.k.a. "How to Not Summon Skynet")

### Step 1 — Install Dependencies

```bash
pip install pandas numpy yfinance fredapi statsmodels openpyxl matplotlib python-dateutil
```

### Step 2 — Set Your FRED API Key

Edit:

```python
FRED_API_KEY = "YOUR_REAL_KEY_HERE"
```

### Step 3 — Run the Main Script

```bash
python full_assignment.py
```

### Step 4 — Copy Output into Your Capstone PDF

- Excel sheets
- Graphs
- WACC table
- NPV/IRR section

---

## 6. Notes on D/E & Tax Rate

### D/E Ratio
- Pulled automatically from Yahoo
- Script determines whether Yahoo is returning a percent or a ratio
- User can override at runtime

### Tax Rate
- Ideally taken from the firm's 10-K
- Enter as a decimal (e.g., 0.21)

---

## 7. Disclaimer from the Office of Algorithmic Overreach

### This codebase does **not**:
- Launch missiles
- Self-modify
- Claim a 35% crypto return is normal (thankfully removed)

### This codebase **does**:
- Pull live financial data
- Produce fully documented betas, WACC, NPV, IRR
- Provide enough quantitative firepower for an MBA capstone
- Sound vaguely like Skynet trying to be helpful

---

> *"The future is not set.*  
> *There is no fate but what we make for ourselves…*  
> *…and what `full_assignment_3.py` writes to Excel."*
