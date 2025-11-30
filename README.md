# Crop Risk Analysis by Region

Agricultural crop risk analysis project focusing on price and yield forecasting for wheat, barley, and maize in Konya region.

## Data Sources
- **Weather**: ERA5 (temperature), CHIRPS (precipitation)
- **Prices**: TURİB daily prices, FAOSTAT
- **Yield**: TÜİK (Turkish Statistical Institute)

## Structure
```
├── data/
│   ├── raw/           # Source data
│   ├── interim/       # Preprocessed data
│   └── processed/     # Model-ready datasets
├── src/
│   ├── data/          # Data processing modules
│   ├── features/      # Feature engineering
│   ├── models/        # Forecasting & simulation models
│   └── visualization/ # Plotting utilities
├── scripts/           # Pipeline execution scripts
└── reports/           # Analysis outputs
```

## Setup
```bash
pip install -r requirements.txt
```
