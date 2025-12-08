# Finance Capstone — Beta Calculation & Data Pipeline

This repository contains a collection of scripts developed for a finance capstone project centered on calculating CAPM beta, pulling financial data, and resolving date-alignment issues across datasets.

The project explored multiple approaches, but ultimately **`date_issue_fix.py`** emerged as the correct and complete solution. The other scripts remain for documentation, experimentation, and future development.

---

## 📌 Repository Overview

### ✔ `date_issue_fix.py` — Final Working Solution
This is the primary script and the end result of the development process. It:

- Aligns daily and monthly financial time series
- Fixes date parsing and resampling errors
- Retrieves market, stock, and risk-free rates (via FRED API)
- Calculates CAPM beta using:
  - Standard OLS regression
  - Zero-intercept regression
- Produces charts, diagnostics, and clean output

**This is the script to run for accurate beta calculations.**

---

### 🚧 `beta_lab_streamlit.py` — Work in Progress
An early prototype of a Streamlit front-end that will eventually:

- Allow user input for tickers, date ranges, and risk-free rates  
- Display regression charts  
- Show formulas and beta calculations  
- Offer choice of market indices  

This app is not fully functional yet but remains in active development.

---

## 🧪 Exploratory & Supporting Scripts

### `calculate_returns.py`
Used to explore:
- Date parsing issues
- Daily and monthly return calculations
- Intermediate data cleaning steps

Helpful during early experimentation but not required in the final workflow.

---

### `CAPM.py`
An exploratory CAPM implementation containing:
- Expected return formulas  
- Covariance & variance mechanics  
- A basic beta calculation  

Useful for understanding underlying theory, but replaced by the final engine.

---

### `pull_t_bills.py`
Prototype script for retrieving Treasury bill yields from FRED.

Later replaced by improved risk-free rate handling in `date_issue_fix.py`.

---

### `pull_cigna_price.py` & `pull_pfizer.py`
Early experiments for pulling specific stock prices.

Cigna prices ended up being unnecessary for the final analysis, but the scripts were kept for reference.


