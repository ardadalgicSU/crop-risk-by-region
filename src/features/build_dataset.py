"""
Build modeling-ready feature matrices from processed P/Y data.

This keeps feature engineering separate from raw data loading so that
notebooks can simply read `data/processed/features_*.csv`.

Outputs (examples):
  - data/processed/features_Y-wheat.csv
  - data/processed/features_P-wheat-monthly.csv
"""

from pathlib import Path

import pandas as pd


PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def build_yield_feature_matrix(crop: str) -> Path:
    """
    Use Y-{crop}-highfreq as source and expose yield_t_ha as the target.
    All other columns (including year and climate/hazard drivers) are features.
    """
    src = PROCESSED / f"Y-{crop}-highfreq.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    if "yield_t_ha" not in df.columns:
        raise ValueError(f"'yield_t_ha' not found in {src}")

    y = df["yield_t_ha"].copy()
    X = df.drop(columns=["yield_t_ha"])
    X["target_yield_t_ha"] = y

    out_path = PROCESSED / f"features_Y-{crop}.csv"
    X.to_csv(out_path, index=False)
    return out_path


def build_price_feature_matrix_monthly(crop: str) -> Path:
    """
    Use P-{crop}-monthly as source and create a simple TS feature set:
      - target: price_real
      - features: price_real_lag1/lag2, volume, usdtry, month dummies.
    """
    src = PROCESSED / f"P-{crop}-monthly.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    required = {"year", "month", "price_real"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {src}: {missing}")

    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
    )

    # Lag features on real price
    df["price_real_lag1"] = df["price_real"].shift(1)
    df["price_real_lag2"] = df["price_real"].shift(2)

    # Month dummies to capture seasonality
    month_dummies = pd.get_dummies(df["month"].astype(int), prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)

    # Target
    df["target_price_real"] = df["price_real"]

    # Drop early rows with NaN lags
    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "price_real",
            "year",
            "date",
        }
    ]
    df = df.dropna(subset=["price_real_lag1", "price_real_lag2"]).reset_index(drop=True)

    out = df[["date"] + feature_cols]
    out_path = PROCESSED / f"features_P-{crop}-monthly.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main():
    # Build yield feature matrices for all three crops
    for crop in ["wheat", "barley", "maize"]:
        try:
            path = build_yield_feature_matrix(crop)
            print(f"Wrote {path}")
        except FileNotFoundError:
            print(f"Skip yield features for {crop} (source missing)")

    # Build monthly price feature matrices
    for crop in ["wheat", "barley", "maize"]:
        try:
            path = build_price_feature_matrix_monthly(crop)
            print(f"Wrote {path}")
        except FileNotFoundError:
            print(f"Skip price features for {crop} (source missing)")


if __name__ == "__main__":
    main()
