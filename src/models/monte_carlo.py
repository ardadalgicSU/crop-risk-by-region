from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class SimulationResult:
    crop: str
    simulations: pd.DataFrame
    summary: Dict[str, float]


def _resolve_data_path(data_dir: Path, filename: str) -> Path:
    candidates = [
        data_dir / filename,
        data_dir / "processed" / filename,
        data_dir / "interim" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename} not found under {data_dir}")


def _load_price_monthly(crop: str, data_dir: Path) -> pd.DataFrame:
    path = _resolve_data_path(data_dir, f"P-{crop}-monthly.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df.sort_values("date").reset_index(drop=True)


def _fit_price_ets(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    seasonal_periods = 12
    seasonal = "add" if len(series) >= seasonal_periods * 2 else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
    )
    fitted = model.fit(optimized=True)
    fitted_values = fitted.fittedvalues
    residuals = (series - fitted_values).dropna().values
    forecast = fitted.forecast(12).values
    return residuals, forecast


def _load_yield_features(crop: str, data_dir: Path) -> pd.DataFrame:
    path = _resolve_data_path(data_dir, f"features_Y-{crop}.csv")
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError("Yield features must include a 'year' column")
    return df.sort_values("year").reset_index(drop=True)


def _prepare_yield_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = "production_mass_t"
    if target not in df.columns:
        if {"yield_t_ha", "harvest_area_ha"}.issubset(df.columns):
            df = df.copy()
            df[target] = df["yield_t_ha"] * df["harvest_area_ha"]
        else:
            raise ValueError("Missing production_mass_t or yield_t_ha/harvest_area_ha")

    drop_cols = {"target_yield_t_ha", "yield_t_ha", "production_mass_t", "crop_name"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.ffill().bfill()
    X = X.fillna(X.mean(numeric_only=True))
    y = df[target].astype(float)
    return X, y


def _fit_yield_model(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    X, y = _prepare_yield_xy(df)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X, y)
    fitted = model.predict(X)
    residuals = y.values - fitted

    x_next = X.iloc[-1].copy()
    if "year" in x_next.index:
        x_next["year"] = x_next["year"] + 1
    x_next = x_next.to_frame().T
    x_next = x_next.apply(pd.to_numeric, errors="coerce")
    x_next = x_next.ffill().bfill().fillna(x_next.mean(numeric_only=True))
    next_pred = model.predict(x_next)[0]
    return residuals, next_pred, y


def _summarize_simulation(df: pd.DataFrame) -> Dict[str, float]:
    earnings = df["earnings"]
    p5 = earnings.quantile(0.05)
    cvar5 = earnings[earnings <= p5].mean()
    return {
        "expected_earnings": float(earnings.mean()),
        "earnings_std": float(earnings.std()),
        "earnings_p5": float(p5),
        "earnings_p95": float(earnings.quantile(0.95)),
        "earnings_cvar_5": float(cvar5),
        "price_mean_mean": float(df["price_mean"].mean()),
        "price_mean_std": float(df["price_mean"].std()),
        "production_mass_mean": float(df["production_mass_t"].mean()),
        "production_mass_std": float(df["production_mass_t"].std()),
    }


def run_simulation(
    crop: str,
    n_sims: int = 10000,
    seed: int = 42,
    data_dir: Path = Path("data"),
) -> SimulationResult:
    rng = np.random.default_rng(seed)

    price_df = _load_price_monthly(crop, data_dir)
    residuals_p, forecast = _fit_price_ets(price_df["price_real"])

    yield_df = _load_yield_features(crop, data_dir)
    residuals_y, next_prod, _ = _fit_yield_model(yield_df)

    price_resid = rng.choice(residuals_p, size=(n_sims, 12), replace=True)
    price_paths = forecast.reshape(1, -1) + price_resid
    price_paths = np.clip(price_paths, 0.0001, None)
    price_mean = price_paths.mean(axis=1)

    prod_resid = rng.choice(residuals_y, size=n_sims, replace=True)
    production_mass = next_prod + prod_resid
    production_mass = np.clip(production_mass, 0.0, None)

    earnings = price_mean * production_mass

    sims = pd.DataFrame(
        {
            "sim_id": np.arange(1, n_sims + 1),
            "price_mean": price_mean,
            "production_mass_t": production_mass,
            "earnings": earnings,
        }
    )
    summary = _summarize_simulation(sims)
    return SimulationResult(crop=crop, simulations=sims, summary=summary)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo simulation with residual bootstrap.")
    parser.add_argument(
        "--crops",
        default="wheat,barley",
        help="Comma-separated crops to simulate (default: wheat,barley).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=10000,
        help="Number of simulations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/predictions",
        help="Directory to write simulation outputs.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crops = [c.strip() for c in args.crops.split(",") if c.strip()]
    summary_rows: List[Dict[str, float]] = []

    for crop in crops:
        result = run_simulation(crop=crop, n_sims=args.sims, seed=args.seed, data_dir=data_dir)
        sim_path = output_dir / f"monte_carlo_{crop}.csv"
        result.simulations.to_csv(sim_path, index=False)

        summary = {"crop": crop}
        summary.update(result.summary)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "monte_carlo_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
