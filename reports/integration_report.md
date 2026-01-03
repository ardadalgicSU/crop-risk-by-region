# Integration (P × Y) Report

**Setup**
- Target: earnings = price_mean * production_mass_t
- Holdout: last 2 years
- Model: Bayesian Ridge
- Variants: full features, minimal (price_mean + production_mass_t), full_log1p

**Best Variant per Crop (lowest RMSE)**

| crop | variant | MAE | RMSE | SMAPE | R² |
| --- | --- | --- | --- | --- | --- |
| barley | full | 94811.800 | 100914.468 | 7.084 | 0.451 |
| maize | full | 124571.123 | 130815.412 | 4.688 | -31.375 |
| wheat | minimal | 70739.034 | 71090.101 | 3.093 | 0.620 |

**Finding**
- Maize integration is unstable: all variants show very low/negative R² on a 2-year holdout, likely due to the short time span (2019–2024) and higher production volatility. Treat maize earnings forecasts as unreliable until more years or external data are added.