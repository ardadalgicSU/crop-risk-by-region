"""
Rebuild Y-band feature tables using growing-season aggregates.

Backs up existing Y-*-highfreq.csv and features_Y-*.csv to data/old/
before writing new versions into data/.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd


DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"

CLIMATE_DAILY = INTERIM_DIR / "climate_master_data.csv"
HAZARDS_MONTHLY = INTERIM_DIR / "climate_hazards_monthly.csv"
YIELD_MASTER = INTERIM_DIR / "yield_master.csv"


@dataclass(frozen=True)
class CropConfig:
    gs_months: list[int]
    shift_months: list[int]


CROP_CONFIG: dict[str, CropConfig] = {
    "wheat": CropConfig(gs_months=[10, 11, 12, 1, 2, 3, 4, 5, 6], shift_months=[10, 11, 12]),
    "barley": CropConfig(gs_months=[10, 11, 12, 1, 2, 3, 4, 5, 6], shift_months=[10, 11, 12]),
    "maize": CropConfig(gs_months=[4, 5, 6, 7, 8, 9], shift_months=[]),
}


def backup_file(path: Path, backup_dir: Path) -> None:
    if not path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_{timestamp}{path.suffix}"
    shutil.copy2(path, backup_dir / backup_name)


def build_climate_monthly() -> pd.DataFrame:
    if not CLIMATE_DAILY.exists():
        raise FileNotFoundError(CLIMATE_DAILY)
    climate = pd.read_csv(CLIMATE_DAILY, parse_dates=["date"])
    climate["year"] = climate["date"].dt.year
    climate["month"] = climate["date"].dt.month
    monthly = (
        climate.groupby(["year", "month"], as_index=False)
        .agg(
            {
                "t2m_min": "mean",
                "t2m_max": "mean",
                "t2m_mean": "mean",
                "precipitation": "sum",
            }
        )
        .rename(columns={"precipitation": "precip_mm"})
    )
    return monthly


def aggregate_gs_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t2m_range"] = df["t2m_max"] - df["t2m_min"]
    agg = (
        df.groupby("gs_year", as_index=False)
        .agg(
            gs_t2m_min=("t2m_min", "mean"),
            gs_t2m_max=("t2m_max", "mean"),
            gs_t2m_mean=("t2m_mean", "mean"),
            gs_t2m_range=("t2m_range", "mean"),
            gs_t2m_std=("t2m_mean", "std"),
            gs_precip_sum=("precip_mm", "sum"),
            gs_precip_mean=("precip_mm", "mean"),
            gs_heatwave_35_days=("heatwave_35", "sum"),
            gs_heatwave_30_days=("heatwave_30", "sum"),
            gs_frost_days=("frost", "sum"),
            gs_heavy_rain_days=("heavy_rain", "sum"),
            gs_dry_spell_max=("dry_spell_max", "max"),
            gs_flood_risk_months=("flood_risk", "sum"),
        )
        .rename(columns={"gs_year": "year"})
    )
    return agg


def build_yield_features(crop: str, climate_full: pd.DataFrame) -> pd.DataFrame:
    if not YIELD_MASTER.exists():
        raise FileNotFoundError(YIELD_MASTER)

    cfg = CROP_CONFIG[crop]
    climate = climate_full[climate_full["month"].isin(cfg.gs_months)].copy()

    climate["gs_year"] = climate["year"]
    if cfg.shift_months:
        climate.loc[climate["month"].isin(cfg.shift_months), "gs_year"] = (
            climate.loc[climate["month"].isin(cfg.shift_months), "year"] + 1
        )

    climate = climate.drop(columns=["year"])

    climate_agg = aggregate_gs_features(climate)

    y = pd.read_csv(YIELD_MASTER, parse_dates=["date"])
    y["year"] = y["date"].dt.year
    y_crop = (
        y[y["crop_name"] == crop]
        .groupby("year", as_index=False)
        .agg(
            {
                "harvest_area_ha": "sum",
                "plant_area_ha": "sum",
                "production_mass_t": "sum",
                "yield_t_ha": "mean",
            }
        )
    )

    out = y_crop.merge(climate_agg, on="year", how="left").sort_values("year")
    return out


def main() -> None:
    backup_dir = DATA_DIR / "old"
    for crop in CROP_CONFIG:
        backup_file(DATA_DIR / f"Y-{crop}-highfreq.csv", backup_dir)
        backup_file(DATA_DIR / f"features_Y-{crop}.csv", backup_dir)

    climate_monthly = build_climate_monthly()
    hazards = pd.read_csv(HAZARDS_MONTHLY)
    climate_full = climate_monthly.merge(hazards, on=["year", "month"], how="left")

    hazard_cols = [
        "heatwave_35",
        "heatwave_30",
        "frost",
        "heavy_rain",
        "dry_spell_max",
        "flood_risk",
    ]
    for col in hazard_cols:
        if col in climate_full.columns:
            climate_full[col] = climate_full[col].fillna(0)

    for crop in CROP_CONFIG:
        out = build_yield_features(crop, climate_full)
        highfreq_path = DATA_DIR / f"Y-{crop}-highfreq.csv"
        out.to_csv(highfreq_path, index=False)

        features = out.copy()
        features["target_yield_t_ha"] = features["yield_t_ha"]
        features_path = DATA_DIR / f"features_Y-{crop}.csv"
        features.to_csv(features_path, index=False)

        print(f"Wrote {highfreq_path} and {features_path}")


if __name__ == "__main__":
    main()
