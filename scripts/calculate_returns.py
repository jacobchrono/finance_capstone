import pandas as pd
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("data")

FILES = [
    "cigna_group_daily_stock_data.csv",
    "cigna_group_monthly_stock_data.csv",
    "snp500_daily_data.csv",
    "snp500_monthly_data.csv",
    "pfizer_daily_stock_data.csv",
    "pfizer_monthly_stock_data.csv"
]

# -----------------------------
# FUNCTION TO CALCULATE RETURNS
# -----------------------------
def calculate_returns(df, name="unknown"):
    """
    Calculate simple returns based on the Close column:
    return_t = (Close_t - Close_(t-1)) / Close_(t-1)
    """
    df = df.copy()

    # Make sure Close is numeric
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found in {name}")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Optional: quick sanity check
    if df["Close"].isna().all():
        raise ValueError(f"All 'Close' values are NaN in {name} after conversion.")

    # Daily / monthly simple returns
    df["Return"] = df["Close"].pct_change(fill_method=None)

    return df


# -----------------------------
# MAIN
# -----------------------------
def main():
    for filename in FILES:
        file_path = DATA_DIR / filename
        print(f"Processing: {file_path}")

        # Read CSV, skipping rows 1 and 2 (0-based indices in the file)
        # Row 0 is still treated as the header.
        df = pd.read_csv(file_path, skiprows=[1, 2])

        # Rename the first column to 'Date'
        first_col_name = df.columns[0]
        df = df.rename(columns={first_col_name: "Date"})

        # Parse Date column and set as index (if it parses)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")

        # Sort by date
        df = df.sort_index()

        # Calculate returns
        df_with_returns = calculate_returns(df, name=filename)

        # Save output
        new_filename = filename.replace(".csv", "_with_returns.csv")
        output_path = DATA_DIR / new_filename
        df_with_returns.to_csv(output_path)

        print(f"Saved → {output_path}\n")


if __name__ == "__main__":
    main()
