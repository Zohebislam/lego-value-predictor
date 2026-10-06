"""Step 1: Load the two Excel files, compute the target, and engineer release-day features.

DATA: Dobrynskaya (2021), "LEGO secondary market price data", Mendeley Data,
doi:10.17632/v9hhs66vm3.1. Prices are averages of recent completed eBay sales (via BrickPicker).
  - Dec 2015 file: 2,322 sets with retail price, theme, pieces, minifigs, and the sealed ("new")
    resale price in Dec 2015.
  - Jan 2018 - Apr 2019 file: monthly resale prices for the same sets (we use April 2019).

TARGET: annual growth rate (CAGR) from retail price to resale price:
    (resale price / retail price) ** (1 / years since release) - 1
Annualizing makes a 2003 set and a 2011 set comparable.

LEAKAGE RULE: every model feature must be knowable on release day.

Output: data/processed/lego_model_data.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW, OUT = Path("data/raw"), Path("data/processed")
MIN_YEAR = 1995      # very old sets are few and their retail prices are unreliable
MIN_RRP = 5.0        # skip polybags and promos with near-zero retail prices
TOP_THEMES = 25      # rarer themes are grouped into "Other"

# Themes based on outside franchises (LEGO pays for the license).
# The LEGO Movie is counted as licensed because it was a Warner Bros. film.
LICENSED = {
    "Star Wars", "Harry Potter", "Super Heroes", "Batman", "Lord of the Rings", "Cars",
    "Teenage mutant ninja turtle", "Indiana Jones", "Toy story", "Pirates of Caribean",
    "Spongebob Squarepants", "Minecraft", "Lone Ranger", "Spider-man", "Disney princess",
    "Discovery", "Prince of Persia", "The Simpsons", "The Lego movie",
}


def find(pattern: str) -> Path:
    """Find a file in data/raw by pattern, so spaces vs. underscores in names don't matter."""
    matches = sorted(RAW.glob(pattern))
    if not matches:
        raise SystemExit(f"Couldn't find a file matching '{pattern}' in data/raw/")
    return matches[0]


def main():
    df = pd.read_excel(find("*whole*Dec*2015*.xlsx"), sheet_name="DATA").rename(columns={
        "theme": "theme", "name": "name", "year of release": "release_year",
        "# of pieces": "pieces", "# of minifigures": "minifigs",
        "Secondary market prices of new sets in 2015": "value_new_2015",
        "Secondary market prices of used sets in 2015": "value_used_2015",
        "Primary market price at release": "rrp_usd",
        "Size group (1 - Biggest; 4 - Smallest)": "size_group",
    }).drop(columns=["age"])
    df["theme"] = df["theme"].replace({"Legends of China": "Legends of Chima"})  # typo in source

    monthly = pd.read_excel(find("*whole*2018*2019*.xlsx"), header=1)
    last_col = monthly.columns[-1]  # April 2019
    df = df.merge(monthly[["id", last_col]].rename(columns={last_col: "value_new_2019"}), on="id", how="left")

    n0 = len(df)
    df = df[(df["release_year"] >= MIN_YEAR) & (df["rrp_usd"] >= MIN_RRP) & (df["pieces"] > 0)].copy()

    # ---- targets ----
    # Values are measured in Dec 2015 and Apr 2019; sets are assumed released mid-year.
    df["years_to_2015"] = 2015.95 - (df["release_year"] + 0.5)
    df["years_to_2019"] = 2019.25 - (df["release_year"] + 0.5)
    for yr in ["2015", "2019"]:
        v = df[f"value_new_{yr}"].where(df[f"value_new_{yr}"] > 0)  # 0 means "no sales recorded"
        df[f"cagr_{yr}"] = (v / df["rrp_usd"]) ** (1 / df[f"years_to_{yr}"]) - 1

    # ---- release-day features ----
    df["minifigs"] = df["minifigs"].fillna(0)
    df["log_pieces"] = np.log(df["pieces"])
    df["log_rrp"] = np.log(df["rrp_usd"])
    df["price_per_piece"] = df["rrp_usd"] / df["pieces"]
    df["minifigs_per_100_pieces"] = 100 * df["minifigs"] / df["pieces"]
    df["is_licensed"] = df["theme"].isin(LICENSED).astype(int)
    top = df["theme"].value_counts().index[:TOP_THEMES]
    df["theme_grouped"] = df["theme"].where(df["theme"].isin(top), "Other")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "lego_model_data.csv", index=False)
    print(f"{n0:,} sets -> {len(df):,} after cleaning "
          f"({df['cagr_2015'].notna().sum():,} with a 2015 value, {df['cagr_2019'].notna().sum():,} with a 2019 value)")


if __name__ == "__main__":
    main()
