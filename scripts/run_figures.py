from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.visualization.make_report_figures import main  # noqa: E402


if __name__ == "__main__":
    main()
