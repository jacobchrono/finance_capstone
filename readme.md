**Department of Defense – Office of Algorithmic Overreach**  
**Subject:** Why Let Skynet Price Risk When You Can Do It Yourself?

**Jacob Clement**  
Created with assistance from ChatGPT and Claude 
https://chatgpt.com/share/6937a88e-eb04-800a-8033-44a9a2056cc9

https://claude.ai/share/41a576df-e0fe-4e23-a8e3-6ad43ca84ff8

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
