from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.price_model import run_all_models, summarize_metrics  # noqa: E402


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "reports" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for crop in ["wheat", "barley", "maize"]:
        results = run_all_models(crop=crop, data_dir=data_dir)
        metrics_df = summarize_metrics(results.values())
        metrics_df.insert(0, "crop", crop)
        all_rows.append(metrics_df)

        metrics_df.to_csv(output_dir / f"price_metrics_{crop}.csv", index=False)

    summary = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(columns=["crop", "model", "mae", "rmse", "smape"])
    )
    summary.to_csv(output_dir / "price_metrics_all.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
