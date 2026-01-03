from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.monte_carlo import run_simulation  # noqa: E402


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "reports" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for crop in ["wheat", "barley"]:
        result = run_simulation(crop=crop, n_sims=10000, seed=42, data_dir=data_dir)
        result.simulations.to_csv(output_dir / f"monte_carlo_{crop}.csv", index=False)

        summary = {"crop": crop}
        summary.update(result.summary)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "monte_carlo_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    import pandas as pd

    main()
