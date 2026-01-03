from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ModelResult:
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    test_index: pd.Series
    metrics: Dict[str, float]
    model: object


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": _smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


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


def load_price_yearly(crop: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    path = _resolve_data_path(data_dir, f"P-{crop}-monthly.csv")
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError("P-monthly file must include a 'year' column")

    grouped = (
        df.groupby("year", as_index=False)
        .agg(
            price_mean=("price_real", "mean"),
            price_std=("price_real", "std"),
            price_min=("price_real", "min"),
            price_max=("price_real", "max"),
            volume_sum=("volume", "sum"),
        )
        .sort_values("year")
    )
    grouped["price_cv"] = grouped["price_std"] / grouped["price_mean"].replace(0, np.nan)
    return grouped


def load_yield_features(crop: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    path = _resolve_data_path(data_dir, f"features_Y-{crop}.csv")
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError("Yield features must include a 'year' column")
    df = df.sort_values("year")
    df = df.drop(columns=["target_yield_t_ha"], errors="ignore")
    return df


def build_integration_dataset(crop: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    price_yearly = load_price_yearly(crop, data_dir=data_dir)
    yield_features = load_yield_features(crop, data_dir=data_dir)

    merged = price_yearly.merge(yield_features, on="year", how="inner")

    if "production_mass_t" not in merged.columns:
        if {"yield_t_ha", "harvest_area_ha"}.issubset(merged.columns):
            merged["production_mass_t"] = merged["yield_t_ha"] * merged["harvest_area_ha"]
        else:
            raise ValueError("Missing production_mass_t or yield_t_ha/harvest_area_ha")

    merged["target_earnings"] = merged["price_mean"] * merged["production_mass_t"]
    return merged.sort_values("year").reset_index(drop=True)


def _prepare_features(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series]:
    y = df["target_earnings"].astype(float)
    if feature_set == "minimal":
        required = ["price_mean", "production_mass_t"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X = df[required].copy()
    else:
        X = df.drop(columns=["target_earnings"])
        X = X.drop(columns=["year"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.ffill().bfill()
    X = X.fillna(X.mean(numeric_only=True))
    return X, y


def _transform_target(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    if transform is None:
        return y
    if transform == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unknown transform: {transform}")


def _inverse_transform_target(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    if transform is None:
        return y
    if transform == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unknown transform: {transform}")


def fit_bayesian_ridge(
    crop: str,
    test_size: int = 2,
    data_dir: Path = Path("data"),
    feature_set: str = "full",
    target_transform: Optional[str] = None,
) -> ModelResult:
    df = build_integration_dataset(crop, data_dir=data_dir)
    if test_size >= len(df):
        raise ValueError("test_size must be smaller than number of rows")

    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()

    X_train, y_train_raw = _prepare_features(train_df, feature_set=feature_set)
    X_test, y_test_raw = _prepare_features(test_df, feature_set=feature_set)

    model = BayesianRidge()
    y_train = _transform_target(y_train_raw.values, target_transform)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = _inverse_transform_target(preds, target_transform)

    metrics = _compute_metrics(y_test_raw.values, preds)
    model_name = "bayesian_ridge"
    if feature_set == "minimal":
        model_name += "_minimal"
    if target_transform:
        model_name += f"_{target_transform}"
    return ModelResult(
        model_name=model_name,
        y_true=y_test_raw.values,
        y_pred=np.asarray(preds),
        test_index=test_df["year"],
        metrics=metrics,
        model=model,
    )


def summarize_metrics(results: List[ModelResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {"model": res.model_name}
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train integration model (Bayesian Ridge).")
    parser.add_argument(
        "--crop",
        default="wheat",
        choices=["wheat", "barley", "maize"],
        help="Crop to model.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2,
        help="Number of years for the test split.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        choices=["full", "minimal"],
        help="Feature set to use.",
    )
    parser.add_argument(
        "--target-transform",
        default="none",
        choices=["none", "log1p"],
        help="Optional target transform.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/predictions",
        help="Directory to write predictions and metrics.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    transform = None if args.target_transform == "none" else args.target_transform
    result = fit_bayesian_ridge(
        crop=args.crop,
        test_size=args.test_size,
        data_dir=data_dir,
        feature_set=args.feature_set,
        target_transform=transform,
    )
    metrics_df = summarize_metrics([result])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / f"integration_metrics_{args.crop}.csv", index=False)

    preds = pd.DataFrame(
        {
            "year": result.test_index,
            "y_true": result.y_true,
            "y_pred": result.y_pred,
            "model": result.model_name,
        }
    )
    preds.to_csv(output_dir / f"integration_predictions_{args.crop}.csv", index=False)

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
