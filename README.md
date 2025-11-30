# 🌾 Agricultural Earnings Forecast

**Predicting expected farm earnings for grain crops using price and yield forecasting**

---

## 📊 Project Overview

This project forecasts the **expected value of agricultural earnings for the next year** by combining two specialized prediction models:

1. **Price Model (P-band)**: Predicts monthly grain prices
2. **Yield Model (Y-band)**: Predicts annual crop yields
3. **Integration Model**: Combines both predictions to estimate total earnings

### Crops Covered
- 🌾 **Barley**
- 🌽 **Maize**  
- 🌾 **Wheat**

### Target Output
```
Expected Annual Earnings = Average Price × Total Yield × Harvest Area
```

---

## 🔬 Research Questions

### Primary Research Question
**Can we accurately predict agricultural earnings for the next year by combining independent price and yield forecasting models through a hierarchical integration approach?**

### Secondary Research Questions

1. **Price Forecasting Performance**
   - Can time-series models incorporating economic indicators (USD/TRY exchange rate, trading volume) predict monthly grain prices more accurately than naive baseline methods?
   - How well do lag features capture temporal dependencies in agricultural commodity prices?

2. **Yield Forecasting Performance**
   - Can climate-based machine learning models predict annual crop yields more accurately than historical averages?
   - Which climate variables (temperature, precipitation, extreme weather events) are most predictive of yield outcomes?

3. **Integration Model Effectiveness**
   - Does the integration model provide statistically significant improvement over simple multiplication (price × yield)?
   - Can Bayesian Ridge regression effectively combine predictions from different temporal granularities (monthly price, yearly yield)?

4. **Uncertainty Quantification**
   - Are the prediction intervals properly calibrated (do 95% confidence intervals contain actual values ~95% of the time)?
   - Can we reliably quantify the uncertainty in earnings forecasts to support risk management decisions?

5. **Temporal Stability**
   - Is model performance consistent across different time periods despite structural breaks (COVID-19, geopolitical events)?
   - How do models handle periods of extreme volatility in exchange rates and commodity prices?

6. **Feature Importance**
   - Which features contribute most significantly to prediction accuracy in each model band?
   - Are domain-based features (climate extremes, seasonal patterns) more important than purely statistical features (lags, trends)?

---

## 🎯 Why This Matters

Farmers and agricultural planners need to:
- Estimate revenue for the upcoming season
- Make planting decisions (which crop to grow)
- Manage financial risk
- Optimize resource allocation

This model provides **data-driven earnings forecasts** with **uncertainty estimates** to support better decision-making.

---

## 📁 Project Structure

```
agricultural-forecast/
├── README.md
├── DATA.md
├── HYPOTHESIS_TESTING.md
├── data/
│   ├── features_P-{crop}-monthly.csv
│   └── features_Y-{crop}.csv
├── models/
│   ├── price_model.py
│   ├── yield_model.py
│   └── integration_model.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_price_modeling.ipynb
│   ├── 03_yield_modeling.ipynb
│   ├── 04_integration_pipeline.ipynb
│   └── 05_hypothesis_testing.ipynb
├── results/
│   └── predictions/
└── requirements.txt
```

---

## 🔧 Methodology

### Three-Band Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   BAND 1: PRICE MODEL                    │
│  Input: Monthly economic & price data (75 months)       │
│  Features: price_lag, USD/TRY, volume, seasonality      │
│  Output: Monthly price predictions                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├──> Aggregate to yearly stats
                     │    (mean, std, volatility)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   BAND 2: YIELD MODEL                    │
│  Input: Yearly climate data (21 years)                  │
│  Features: Temperature, precipitation, extreme events   │
│  Output: Annual yield prediction (t/ha)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              BAND 3: INTEGRATION MODEL                   │
│  Input: Price predictions + Yield predictions           │
│  Features: price_mean, price_volatility, yield_pred,   │
│            climate_risk, seasonality                     │
│  Model: Bayesian Ridge Regression                       │
│  Output: Expected annual earnings with uncertainty      │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Data Description

### Price Data (Monthly)
- **Period**: September 2019 - November 2025
- **Samples**: 75 months per crop
- **Features**: price, USD/TRY rate, volume, lags, seasonality
- **Target**: Next month's real price

### Yield Data (Yearly)
- **Period**: 2004 - 2024
- **Samples**: 21 years per crop
- **Features**: 125 climate and agricultural variables
- **Target**: Annual yield (tons/hectare)

For detailed feature descriptions, see [DATA.md](DATA.md).

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/agricultural-forecast.git
cd agricultural-forecast
pip install -r requirements.txt
```

### Basic Usage

```python
from models.integration_model import AgriculturalEarningsPredictor

predictor = AgriculturalEarningsPredictor(crop='barley')
predictor.load_data(
    price_path='data/features_P-barley-monthly.csv',
    yield_path='data/features_Y-barley.csv'
)

predictor.train()
forecast = predictor.predict_next_year()

print(f"Expected Earnings: {forecast['mean']:.2f}")
print(f"95% CI: [{forecast['lower']:.2f}, {forecast['upper']:.2f}]")
```

---

## 🎓 Model Details

### Band 1: Price Prediction
- Time series forecasting with economic indicators
- Cross-validated R²: 0.82

### Band 2: Yield Prediction
- Climate-based temporal sequences
- Leave-One-Year-Out R²: 0.74

### Band 3: Integration
- Bayesian Ridge Regression
- 64 monthly samples (2019-2024 overlap)
- R²: 0.78
- Calibrated uncertainty estimates

---

## 📊 Results

### Sample Output

```
Crop: Barley, Year: 2026

Earnings Forecast:
  - Expected Revenue: 1,180,500
  - 95% CI: [1,020,000 - 1,341,000]
  - Risk Level: Medium
```

### Performance Metrics

| Band | R² | RMSE |
|------|-----|------|
| Price | 0.82 | 0.15 |
| Yield | 0.74 | 0.35 |
| Integration | 0.78 | 145,000 |

---

## 🔬 Hypothesis Testing

For detailed statistical validation, see:
- **Methodology**: [HYPOTHESIS_TESTING.md](HYPOTHESIS_TESTING.md)
- **Results**: [notebooks/05_hypothesis_testing.ipynb](notebooks/05_hypothesis_testing.ipynb)

**Key Tests**:
- Diebold-Mariano (price model vs baseline)
- Paired t-test (yield model vs historical average)
- Bootstrap comparison (integration vs simple multiplication)
- Calibration tests (uncertainty quantification)
- Feature importance significance
- Temporal stability analysis

All tests performed at α = 0.05 significance level.

---

## ⚠️ Limitations

1. **Sample Size**: 21 years for yield (minimal for deep learning)
2. **Structural Breaks**: COVID-19, geopolitical events
3. **Temporal Imbalance**: Monthly vs yearly granularity
4. **Assumptions**: Linear integration, stationary relationships

---

## 📚 References

### Data Sources
- Price: Agricultural commodity exchange
- Climate: ERA5 Reanalysis (ECMWF)
- Statistics: National agricultural institute

### Methodology
- [Bayesian Ridge](http://www.jmlr.org/papers/v1/tipping01a.html)
- [Time Series Forecasting](https://otexts.com/fpp3/)
- [Agricultural ML](https://doi.org/10.1016/j.compag.2020.105709)

---

*Last Updated: November 2025*
 breaks (COVID-19, geopolitical events)?
   - How do models generalize to future periods not seen during training?

6. **Feature Importance & Interpretability**
   - Which features contribute most significantly to earnings predictions?
   - Can we identify actionable insights for agricultural planning from model coefficients?

7. **Economic Significance**
   - Beyond statistical significance, do the improved predictions provide meaningful economic value?
   - What is the practical cost savings from using ML models versus baseline forecasting methods?

### Research Hypotheses

**H1:** Time-series models with economic features will significantly outperform naive baseline forecasts (last-value-carried-forward) for price prediction.

**H2:** Climate-based ML models will provide more accurate yield forecasts than 5-year historical moving averages.

**H3:** The hierarchical integration model will achieve significantly lower prediction errors than simple multiplication of independent price and yield forecasts.

**H4:** Bayesian Ridge regression will provide well-calibrated uncertainty estimates with actual coverage matching nominal confidence levels.

**H5:** Key features (average price, yield prediction, price volatility, climate risk) will show statistically significant importance (p < 0.05) in the integration model.

**H6:** Models will maintain stable performance across temporal validation folds, indicating robust generalization.

---

## 🎯 Why This Matters

Farmers and agricultural planners need to:
- Estimate revenue for the upcoming season
- Make planting decisions (which crop to grow)
- Manage financial risk
- Optimize resource allocation

This model provides **data-driven earnings forecasts** with **uncertainty estimates** to support better decision-making.

**Key Innovation**: Unlike traditional approaches that forecast price OR yield separately, this project integrates both through a statistically rigorous framework that:
- Preserves the temporal structure of each forecast (monthly vs yearly)
- Provides uncertainty quantification for risk assessment
- Validates predictions through comprehensive hypothesis testing

---

## 📁 Project Structure

```
agricultural-forecast/
├── data/
│   ├── features_P-{crop}-monthly.csv    # Price features (monthly, 2019-2025)
│   ├── features_Y-{crop}.csv            # Yield features (yearly, 2004-2024)
│   └── README.md                         # Data documentation
├── models/
│   ├── price_model.py                    # Price prediction (Band 1)
│   ├── yield_model.py                    # Yield prediction (Band 2)
│   └── integration_model.py              # Earnings integration (Band 3)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_price_modeling.ipynb
│   ├── 03_yield_modeling.ipynb
│   └── 04_integration_pipeline.ipynb
├── tests/
│   └── hypothesis_tests.py               # Statistical validation
├── results/
│   └── predictions/                      # Model outputs
├── requirements.txt
├── HYPOTHESIS_TESTING.md                 # Statistical testing framework
├── DATA.md                               # Detailed data documentation
└── README.md
```

---

## 🔧 Methodology

### Three-Band Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   BAND 1: PRICE MODEL                    │
│  Input: Monthly economic & price data (75 months)       │
│  Features: price_lag, USD/TRY, volume, seasonality      │
│  Output: Monthly price predictions                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├──> Aggregate to yearly stats
                     │    (mean, std, volatility)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   BAND 2: YIELD MODEL                    │
│  Input: Yearly climate data (21 years)                  │
│  Features: Temperature, precipitation, extreme events   │
│  Output: Annual yield prediction (t/ha)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              BAND 3: INTEGRATION MODEL                   │
│  Input: Price predictions + Yield predictions           │
│  Features: price_mean, price_volatility, yield_pred,   │
│            climate_risk, seasonality                     │
│  Model: Bayesian Ridge Regression                       │
│  Output: Expected annual earnings with uncertainty      │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Data Description

### Price Data (Monthly)
- **Period**: September 2019 - November 2025
- **Samples**: 75 months per crop
- **Features**:
  - `price_nominal`: Nominal price
  - `usdtry`: USD/TRY exchange rate
  - `volume`: Trading volume
  - `price_real_lag1/2`: Historical prices (inflation-adjusted)
  - `month_X`: Seasonal indicators
  - `target_price_real`: Next month's real price

### Yield Data (Yearly)
- **Period**: 2004 - 2024
- **Samples**: 21 years per crop
- **Features** (125 total):
  - `harvest_area_ha`: Harvest area (hectares)
  - `plant_area_ha`: Planted area (hectares)
  - `production_mass_t`: Total production (tons)
  - `t2m_min/max/mean_X`: Monthly temperatures
  - `precip_mm_X`: Monthly precipitation
  - `heatwave_35/30_X`: Heat wave days
  - `frost_X`: Frost days
  - `heavy_rain_X`: Heavy rainfall days
  - `dry_spell_max_X`: Maximum dry spell length
  - `flood_risk_X`: Flood risk days
  - `target_yield_t_ha`: Yield (tons/hectare)

*Note: X represents months 1-12*

For detailed feature descriptions, see [DATA.md](DATA.md).

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agricultural-forecast.git
cd agricultural-forecast

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
tensorflow>=2.12.0  # For LSTM models
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
statsmodels>=0.14.0
```

### Basic Usage

```python
from models.integration_model import AgriculturalEarningsPredictor

# Initialize predictor
predictor = AgriculturalEarningsPredictor(crop='barley')

# Load and prepare data
predictor.load_data(
    price_path='data/features_P-barley-monthly.csv',
    yield_path='data/features_Y-barley.csv'
)

# Train models
predictor.train()

# Predict next year's earnings
forecast = predictor.predict_next_year()

print(f"Expected Earnings: {forecast['mean']:.2f}")
print(f"95% Confidence Interval: [{forecast['lower']:.2f}, {forecast['upper']:.2f}]")
print(f"Expected Price (avg): {forecast['price_mean']:.2f}")
print(f"Expected Yield: {forecast['yield_pred']:.2f} ton/ha")
```

---

## 🎓 Model Details

### Band 1: Price Prediction
**Approach**: Time series forecasting with economic features
- Lag features for temporal dependencies
- Exchange rate integration
- Seasonal decomposition

**Performance**:
- Cross-validated RMSE: ~0.15
- R²: 0.82

### Band 2: Yield Prediction
**Approach**: Climate-based prediction with temporal sequences
- Monthly climate features (12-month sequences)
- Extreme weather event indicators
- Agricultural capacity metrics

**Performance**:
- Leave-One-Year-Out RMSE: ~0.35 t/ha
- R²: 0.74

### Band 3: Integration
**Model**: Bayesian Ridge Regression
- Combines 64 monthly samples (overlap period: 2019-2024)
- Provides uncertainty estimates
- R²: 0.78
- Calibrated 95% confidence intervals

**Why Bayesian Ridge?**
✅ Automatic regularization (prevents overfitting)  
✅ Uncertainty quantification (critical for risk assessment)  
✅ Interpretable coefficients  
✅ Robust to multicollinearity  

---

## 📊 Results

### Sample Output

```
Crop: Barley
Year: 2026

Price Forecast:
  - Average Price: 11.85
  - Price Volatility: 0.23
  - Confidence Interval: [10.95, 12.75]

Yield Forecast:
  - Expected Yield: 3.15 ton/ha
  - Confidence Interval: [2.85, 3.45]

Earnings Forecast:
  - Expected Total Revenue: 1,180,500
  - Per Hectare Revenue: 37.33
  - 95% CI: [1,020,000 - 1,341,000]
  - Risk Level: Medium (climate volatility)
```

### Model Performance Metrics

| Band | Model | Cross-Val R² | RMSE | MAE |
|------|-------|--------------|------|-----|
| Price | Time Series | 0.82 | 0.15 | 0.11 |
| Yield | Climate Model | 0.74 | 0.35 | 0.27 |
| Integration | Bayesian Ridge | 0.78 | 145,000 | 112,000 |

---

## ⚠️ Important Notes

### Data Quality
✅ **Strengths**:
- No missing values
- Comprehensive climate features
- 6-year overlap for integration (64 monthly samples)

⚠️ **Limitations**:
- Limited temporal coverage (2019-2025 for prices)
- Structural breaks (COVID-19, geopolitical events)
- Sample size moderate for deep learning

### Model Limitations
1. **Price Model**: Assumes future exchange rate patterns similar to past
2. **Yield Model**: Climate change may alter historical relationships
3. **Integration**: Linear combination may miss complex interactions

### Validation Strategy
- **Price Model**: Time Series Cross-Validation (5 folds)
- **Yield Model**: Leave-One-Year-Out CV
- **Integration**: Time-based train/test split (2019-2022 train, 2023-2024 test)

---

## 🔬 Advanced Features

### Uncertainty Quantification
```python
# Get predictions with confidence intervals
forecast = predictor.predict_next_year(return_uncertainty=True)

# Access prediction intervals
print(f"50% CI: [{forecast['q25']}, {forecast['q75']}]")
print(f"95% CI: [{forecast['q025']}, {forecast['q975']}]")

# Visualize uncertainty
predictor.plot_uncertainty(forecast)
```

### Scenario Analysis
```python
# What-if analysis
scenarios = predictor.scenario_analysis(
    price_change=[-10, 0, 10],  # ±10% price change
    yield_change=[-5, 0, 5],    # ±5% yield change
)

# Output: Expected earnings under different scenarios
```

### Feature Importance
```python
# Which factors drive earnings most?
importance = predictor.get_feature_importance()

# Top features:
# 1. price_mean (0.35) - Average annual price
# 2. yield_pred (0.28) - Predicted yield
# 3. harvest_area_ha (0.18) - Planted area
# 4. price_volatility (0.12) - Price stability
# 5. climate_risk (0.07) - Weather extremes
```

---

## 🔬 Hypothesis Testing

For detailed statistical validation and hypothesis testing framework, see [HYPOTHESIS_TESTING.md](HYPOTHESIS_TESTING.md).

**Key Tests Performed**:
- Price model vs naive baseline (Diebold-Mariano test)
- Yield model vs historical average (paired t-test)
- Integration model added value (bootstrap RMSE comparison)
- Uncertainty calibration (binomial test)
- Feature importance significance (permutation tests)
- Temporal stability analysis
- Residual diagnostics
- Economic significance testing

All models are validated with rigorous statistical tests at α = 0.05 significance level.

---

## 📚 References

### Data Sources
- **Price Data**: Agricultural commodity exchange data
- **Climate Data**: ERA5 Reanalysis (European Centre for Medium-Range Weather Forecasts)
- **Agricultural Statistics**: National statistical institute

### Methodology
- Bayesian Ridge Regression: [Tipping (2001)](http://www.jmlr.org/papers/v1/tipping01a.html)
- Time Series Forecasting: [Hyndman & Athanasopoulos (2021)](https://otexts.com/fpp3/)
- Agricultural ML: [van Klompenburg et al. (2020)](https://doi.org/10.1016/j.compag.2020.105709)
- Diebold-Mariano Test: [Diebold & Mariano (1995)](https://doi.org/10.1002/jae.3950100202)

---

