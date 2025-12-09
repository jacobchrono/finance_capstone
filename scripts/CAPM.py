import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# https://chatgpt.com/share/692fa7b4-3ae0-800a-baf6-22a8ad3d1556

# File paths
CI_FILE = DATA_DIR / "cigna_group_monthly_stock_data.csv"
SP_FILE = DATA_DIR / "snp500_monthly_data.csv"

# -----------------------------------------------------------
# FUNCTION: Load & prepare monthly dataset
# -----------------------------------------------------------
def load_and_clean(path, label):
    df = pd.read_csv(path, skiprows=[1, 2])

    # Standardize date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")

    # Convert Close to numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Monthly returns
    df[f"{label}_Return"] = df["Close"].pct_change(fill_method=None)

    return df[[f"{label}_Return"]]  # return only return column


# -----------------------------------------------------------
# MAIN PROCESSING
# -----------------------------------------------------------
def main():
    print("Loading datasets...")

    ci = load_and_clean(CI_FILE, "CI")
    sp = load_and_clean(SP_FILE, "SP500")

    # Merge on Date index (inner join keeps aligned months)
    merged = ci.join(sp, how="inner")

    print("\nMerged return dataset (first few rows):")
    print(merged.head())

    # ------------------------------------------------------
    # CAPM statistics
    # ------------------------------------------------------
    covariance = merged["CI_Return"].cov(merged["SP500_Return"])
    variance = merged["SP500_Return"].var()

    print("\nCAPM Components:")
    print(f"Covariance(CI, Market) = {covariance}")
    print(f"Variance(Market) = {variance}")

    # Save merged data
    out_path = DATA_DIR / "ci_sp500_monthly_returns_merged.csv"
    merged.to_csv(out_path)
    print(f"\nSaved merged dataset → {out_path}")

    # Save stats
    stats_path = DATA_DIR / "capm_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Covariance(CI, Market): {covariance}\n")
        f.write(f"Variance(Market): {variance}\n")

    print(f"Saved CAPM stats → {stats_path}")


if __name__ == "__main__":
    main()