from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    }


def _train_test_split_ts(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if test_size >= len(df):
        raise ValueError("test_size must be smaller than the number of rows")
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def load_price_monthly(crop: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    path = data_dir / f"P-{crop}-monthly.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_price_features(crop: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    path = data_dir / f"features_P-{crop}-monthly.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fit_sarima(
    crop: str,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
    exog_cols: Optional[Iterable[str]] = None,
    test_size: int = 12,
    data_dir: Path = Path("data"),
) -> ModelResult:
    df = load_price_monthly(crop, data_dir=data_dir)
    train_df, test_df = _train_test_split_ts(df, test_size)

    y_train = train_df["price_real"].astype(float).values
    y_test = test_df["price_real"].astype(float).values

    exog_train = exog_test = None
    if exog_cols:
        exog_cols = list(exog_cols)
        exog_train = train_df[exog_cols].astype(float).values
        exog_test = test_df[exog_cols].astype(float).values

    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    preds = fitted.forecast(steps=test_size, exog=exog_test)

    metrics = _compute_metrics(y_test, np.asarray(preds))
    return ModelResult(
        model_name="sarima",
        y_true=y_test,
        y_pred=np.asarray(preds),
        test_index=test_df["date"],
        metrics=metrics,
        model=fitted,
    )


def fit_ets(
    crop: str,
    seasonal: str = "add",
    trend: str = "add",
    test_size: int = 12,
    data_dir: Path = Path("data"),
) -> ModelResult:
    df = load_price_monthly(crop, data_dir=data_dir)
    train_df, test_df = _train_test_split_ts(df, test_size)

    y_train = train_df["price_real"].astype(float)
    y_test = test_df["price_real"].astype(float).values

    seasonal_periods = 12
    if len(y_train) < seasonal_periods * 2:
        seasonal = None

    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
    )
    fitted = model.fit(optimized=True)
    preds = fitted.forecast(test_size)

    metrics = _compute_metrics(y_test, np.asarray(preds))
    return ModelResult(
        model_name="ets",
        y_true=y_test,
        y_pred=np.asarray(preds),
        test_index=test_df["date"],
        metrics=metrics,
        model=fitted,
    )


def _prepare_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["target_price_real"].astype(float)
    X = df.drop(columns=["target_price_real", "date"])
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    X = X.ffill().bfill()
    return X, y


def fit_ml_baselines(
    crop: str,
    test_size: int = 12,
    data_dir: Path = Path("data"),
) -> Dict[str, ModelResult]:
    df = load_price_features(crop, data_dir=data_dir)
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
            n_estimators=300, random_state=42, n_jobs=-1
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
            test_index=test_df["date"],
            metrics=metrics,
            model=model,
        )

    return results


def summarize_metrics(results: Iterable[ModelResult]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for res in results:
        row = {"model": res.model_name}
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def run_all_models(
    crop: str,
    test_size: int = 12,
    data_dir: Path = Path("data"),
) -> Dict[str, ModelResult]:
    sarima_res = fit_sarima(crop=crop, test_size=test_size, data_dir=data_dir)
    ets_res = fit_ets(crop=crop, test_size=test_size, data_dir=data_dir)
    ml_results = fit_ml_baselines(crop=crop, test_size=test_size, data_dir=data_dir)

    results = {
        "sarima": sarima_res,
        "ets": ets_res,
        **{f"ml_{k}": v for k, v in ml_results.items()},
    }
    return results


def _predictions_table(results: Dict[str, ModelResult]) -> pd.DataFrame:
    frames = []
    for name, res in results.items():
        frames.append(
            pd.DataFrame(
                {
                    "date": res.test_index,
                    "y_true": res.y_true,
                    "y_pred": res.y_pred,
                    "model": name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train P-band price models.")
    parser.add_argument(
        "--crop",
        default="wheat",
        choices=["wheat", "barley", "maize"],
        help="Crop to model.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=12,
        help="Number of months for the test split.",
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
    metrics_df.to_csv(output_dir / f"price_metrics_{args.crop}.csv", index=False)
    _predictions_table(results).to_csv(
        output_dir / f"price_predictions_{args.crop}.csv", index=False
    )

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
