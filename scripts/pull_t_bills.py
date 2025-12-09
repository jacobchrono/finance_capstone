import os
from datetime import datetime
import pandas as pd
from fredapi import Fred

# ---------- CONFIG ----------
START_DATE = "1950-01-01"  # go as far back as possible for most series
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Load FRED API key from environment variable
FRED_API_KEY = os.getenv("FRED_API_KEY")
if FRED_API_KEY is None:
    raise ValueError(
        "FRED_API_KEY environment variable not set. "
        "Set it first, e.g. `setx FRED_API_KEY \"your_key_here\"` in PowerShell."
    )

fred = Fred(api_key=FRED_API_KEY)

# ---------- SERIES TO DOWNLOAD ----------

# Daily Treasury yields (percent per annum)
DAILY_SERIES = {
    "DGS1MO": "1M_Treasury_Yield_D",
    "DGS3MO": "3M_Treasury_Yield_D",
    "DGS6MO": "6M_Treasury_Yield_D",
    "DGS1":   "1Y_Treasury_Yield_D",
    "DGS2":   "2Y_Treasury_Yield_D",
    "DGS5":   "5Y_Treasury_Yield_D",
    "DGS10":  "10Y_Treasury_Yield_D",
    "DGS20":  "20Y_Treasury_Yield_D",
    "DGS30":  "30Y_Treasury_Yield_D",
}

# Monthly Treasury yields (percent per annum)
MONTHLY_SERIES = {
    "TB3MS": "3M_TBill_Yield_M",
    "GS1":   "1Y_Treasury_Yield_M",
    "GS3":   "3Y_Treasury_Yield_M",
    "GS5":   "5Y_Treasury_Yield_M",
    "GS10":  "10Y_Treasury_Yield_M",
    "GS20":  "20Y_Treasury_Yield_M",
    "GS30":  "30Y_Treasury_Yield_M",
}

# ---------- HELPER FUNCTIONS ----------

def fetch_series_to_df(series_map, start_date, end_date, freq=None):
    """
    Fetch multiple FRED series into a single pandas DataFrame.

    Parameters
    ----------
    series_map : dict
        Keys = FRED series IDs, values = column names for the resulting DataFrame.
    start_date : str 'YYYY-MM-DD'
    end_date   : str 'YYYY-MM-DD'
    freq : str or None
        If None, keep native frequency.
        If 'M' or 'D', resample to monthly/daily using mean.

    Returns
    -------
    pandas.DataFrame
        Index = Date, columns = series_map values.
    """
    df_list = []

    for series_id, col_name in series_map.items():
        print(f"Fetching {series_id} -> {col_name} ...")
        s = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date
        )
        s = s.rename(col_name)
        df_list.append(s)

    # Combine into a single DataFrame on the date index
    df = pd.concat(df_list, axis=1)

    # Ensure the index is a proper DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # Optional resample
    if freq is not None:
        # Most yield series are observed daily or monthly; if we
        # resample, taking the mean is a reasonable convention
        df = df.resample(freq).mean()

    return df

# ---------- FETCH DAILY & MONTHLY DATA ----------

print("Downloading DAILY Treasury yield data from FRED...")
daily_yields = fetch_series_to_df(
    DAILY_SERIES,
    start_date=START_DATE,
    end_date=END_DATE,
    freq=None  # native frequency for these is daily
)

print("\nDownloading MONTHLY Treasury yield data from FRED...")
monthly_yields = fetch_series_to_df(
    MONTHLY_SERIES,
    start_date=START_DATE,
    end_date=END_DATE,
    freq=None  # native frequency is monthly
)

# ---------- CLEANUP / SORT / SAVE ----------

daily_yields = daily_yields.sort_index()
monthly_yields = monthly_yields.sort_index()

# Optional: drop rows where all columns are NaN
daily_yields = daily_yields.dropna(how="all")
monthly_yields = monthly_yields.dropna(how="all")

# Save to CSV for Excel / other tools
os.makedirs("data_out", exist_ok=True)
daily_path = os.path.join("data_out", "treasury_yields_daily.csv")
monthly_path = os.path.join("data_out", "treasury_yields_monthly.csv")

daily_yields.to_csv(daily_path, index_label="Date")
monthly_yields.to_csv(monthly_path, index_label="Date")

print(f"\nSaved daily yields to:   {daily_path}")
print(f"Saved monthly yields to: {monthly_path}")
print("\nDone.")