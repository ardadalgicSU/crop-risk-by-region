yok önce bi eğitimleri yapalım # Exploratory Data Analysis (EDA) & Hypothesis Tests

**Purpose:** quick visual checks + explicit null/alternative tests for price, yield, and climate drivers. Keep figures in `reports/figures/` and reference them here.

## 1) Data
- Prices: `data/interim/turib_prices_real_daily.csv`, `data/processed/P-*-monthly.csv`
- Yield: `data/interim/yield_master.csv`, `data/processed/Y-*-highfreq.csv`
- Climate/Hazards: `data/interim/climate_master_data.csv`, `data/interim/climate_hazards_monthly.csv`

## 2) Core Plots (attach PNGs)
- Real price time series (monthly) by crop
- Yield time series (t/ha) by crop
- Price vs USD/TRY scatter (daily, wheat)
- Monthly boxplot (real price seasonality)
- Optional: hazard bars, hazard vs yield/price change, GS anomalies

## 3) Hypothesis Tests (example H0/H1)
1. **Wheat yield: early vs late years (t-test, unequal var)**  
   - H0: mean(early) = mean(late)  
   - H1: mean(early) ≠ mean(late)
2. **USD/TRY vs real price (wheat daily, Spearman)**  
   - H0: rho = 0 (no monotonic association)  
   - H1: rho ≠ 0

## 4) Findings (fill in)
- [Add short bullets after running tests/plots]

## 5) Next Steps
- If specific climate driver is strong, refine features.
- Align plots/tables with slides and final report.
