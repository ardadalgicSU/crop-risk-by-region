from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.integration_model import fit_bayesian_ridge, summarize_metrics  # noqa: E402


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "reports" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        {"name": "full", "feature_set": "full", "target_transform": None},
        {"name": "minimal", "feature_set": "minimal", "target_transform": None},
        {"name": "full_log1p", "feature_set": "full", "target_transform": "log1p"},
    ]

    all_rows = []
    for crop in ["wheat", "barley", "maize"]:
        for variant in variants:
            result = fit_bayesian_ridge(
                crop=crop,
                test_size=2,
                data_dir=data_dir,
                feature_set=variant["feature_set"],
                target_transform=variant["target_transform"],
            )
            metrics_df = summarize_metrics([result])
            metrics_df.insert(0, "crop", crop)
            metrics_df.insert(1, "variant", variant["name"])
            all_rows.append(metrics_df)

            metrics_path = output_dir / f"integration_metrics_{crop}_{variant['name']}.csv"
            metrics_df.to_csv(metrics_path, index=False)

            preds = pd.DataFrame(
                {
                    "year": result.test_index,
                    "y_true": result.y_true,
                    "y_pred": result.y_pred,
                    "model": result.model_name,
                    "variant": variant["name"],
                }
            )
            preds.to_csv(
                output_dir / f"integration_predictions_{crop}_{variant['name']}.csv",
                index=False,
            )

    summary = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(columns=["crop", "variant", "model", "mae", "rmse", "smape", "r2"])
    )
    summary.to_csv(output_dir / "integration_metrics_all.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
