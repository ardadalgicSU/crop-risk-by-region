from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "year" in df.columns:
        return df.sort_values("year").reset_index(drop=True)
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    return df.reset_index(drop=True)


def _train_test_split_ts(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if test_size >= len(df):
        raise ValueError("test_size must be smaller than the number of rows")
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def _build_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    if "yield_t_ha" not in df.columns:
        raise ValueError("'yield_t_ha' not found in the provided dataset")
    out = df.copy()
    out["target_yield_t_ha"] = out["yield_t_ha"]
    return out


def load_yield_features(crop: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    feature_file = f"features_Y-{crop}.csv"
    try:
        path = _resolve_data_path(data_dir, feature_file)
        df = pd.read_csv(path)
        return _ensure_sorted(df)
    except FileNotFoundError:
        pass

    highfreq_file = f"Y-{crop}-highfreq.csv"
    try:
        path = _resolve_data_path(data_dir, highfreq_file)
        df = pd.read_csv(path)
        df = _build_yield_features(df)
        return _ensure_sorted(df)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing features for Y-band. Expected {feature_file} or {highfreq_file}."
        ) from exc


def _prepare_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_col = "target_yield_t_ha" if "target_yield_t_ha" in df.columns else "yield_t_ha"
    if target_col not in df.columns:
        raise ValueError(f"{target_col} missing from dataset")

    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col], errors="ignore")
    X = X.drop(columns=["date", "crop_name"], errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any()]
    X = X.ffill().bfill()
    X = X.fillna(X.mean(numeric_only=True))
    return X, y


def _get_test_index(df: pd.DataFrame) -> pd.Series:
    if "year" in df.columns:
        return df["year"]
    if "date" in df.columns:
        return pd.to_datetime(df["date"])
    return df.index


def fit_ml_models(
    crop: str,
    test_size: int = 4,
    data_dir: Path = Path("data"),
) -> Dict[str, ModelResult]:
    df = load_yield_features(crop, data_dir=data_dir)
    df = _ensure_sorted(df)
    train_df, test_df = _train_test_split_ts(df, test_size)

    X_train, y_train = _prepare_ml_features(train_df)
    X_test, y_test = _prepare_ml_features(test_df)

    models = {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results: Dict[str, ModelResult] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = _compute_metrics(y_test.values, preds)
        results[name] = ModelResult(
            model_name=name,
            y_true=y_test.values,
            y_pred=np.asarray(preds),
            test_index=_get_test_index(test_df),
            metrics=metrics,
            model=model,
        )

    return results


def fit_baseline_mean(
    crop: str,
    test_size: int = 4,
    data_dir: Path = Path("data"),
) -> ModelResult:
    df = load_yield_features(crop, data_dir=data_dir)
    df = _ensure_sorted(df)
    train_df, test_df = _train_test_split_ts(df, test_size)

    target_col = "target_yield_t_ha" if "target_yield_t_ha" in df.columns else "yield_t_ha"
    y_train = train_df[target_col].astype(float).values
    y_test = test_df[target_col].astype(float).values
    mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))

    metrics = _compute_metrics(y_test, mean_pred)
    return ModelResult(
        model_name="baseline_mean",
        y_true=y_test,
        y_pred=mean_pred,
        test_index=_get_test_index(test_df),
        metrics=metrics,
        model=None,
    )


def summarize_metrics(results: Iterable[ModelResult]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for res in results:
        row = {"model": res.model_name}
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def run_all_models(
    crop: str,
    test_size: int = 4,
    data_dir: Path = Path("data"),
) -> Dict[str, ModelResult]:
    baseline = fit_baseline_mean(crop=crop, test_size=test_size, data_dir=data_dir)
    ml_results = fit_ml_models(crop=crop, test_size=test_size, data_dir=data_dir)

    results = {
        "baseline_mean": baseline,
        **{f"ml_{k}": v for k, v in ml_results.items()},
    }
    return results


def _predictions_table(results: Dict[str, ModelResult]) -> pd.DataFrame:
    frames = []
    for name, res in results.items():
        frames.append(
            pd.DataFrame(
                {
                    "index": res.test_index,
                    "y_true": res.y_true,
                    "y_pred": res.y_pred,
                    "model": name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Y-band yield models.")
    parser.add_argument(
        "--crop",
        default="wheat",
        choices=["wheat", "barley", "maize"],
        help="Crop to model.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=4,
        help="Number of years for the test split.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/predictions",
        help="Directory to write predictions and metrics.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results = run_all_models(crop=args.crop, test_size=args.test_size, data_dir=data_dir)
    metrics_df = summarize_metrics(results.values())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / f"yield_metrics_{args.crop}.csv", index=False)
    _predictions_table(results).to_csv(
        output_dir / f"yield_predictions_{args.crop}.csv", index=False
    )

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
