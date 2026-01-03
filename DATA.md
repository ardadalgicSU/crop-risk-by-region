# 📊 Data Documentation

## Overview

This document provides detailed information about the datasets used in the Agricultural Earnings Forecast project.

---

## Dataset Summary

| Dataset | Temporal Resolution | Period | Samples | Features | Size |
|---------|---------------------|--------|---------|----------|------|
| Price (P-barley) | Monthly | 2019-09 to 2025-11 | 75 | 19 | ~13 KB |
| Price (P-maize) | Monthly | 2019-09 to 2025-11 | 75 | 19 | ~13 KB |
| Price (P-wheat) | Monthly | 2019-09 to 2025-11 | 75 | 19 | ~13 KB |
| Yield (Y-barley) | Yearly | 2004 to 2024 | 21 | 125 | ~23 KB |
| Yield (Y-maize) | Yearly | 2004 to 2024 | 21 | 125 | ~23 KB |
| Yield (Y-wheat) | Yearly | 2004 to 2024 | 21 | 125 | ~23 KB |

---

## Price Data Features (P-band)

### File Format
- **Filename Pattern**: `features_P-{crop}-monthly.csv`
- **Delimiter**: Comma (`,`)
- **Encoding**: UTF-8
- **Missing Values**: None

### Feature Descriptions

| Feature | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `date` | datetime | Month (YYYY-MM-DD format) | 2019-09-01 |
| `month` | int | Month number (1-12) | 9 |
| `price_nominal` | float | Nominal price (TRY/ton) | 1.1915 |
| `usdtry` | float | USD/TRY exchange rate | 5.7047 |
| `volume` | float | Trading volume | 3822850.67 |
| `price_real_lag1` | float | Real price 1 month ago (inflation-adjusted) | 1.2102 |
| `price_real_lag2` | float | Real price 2 months ago | 1.1996 |
| `month_2` to `month_12` | boolean | One-hot encoded month indicators | True/False |
| `target_price_real` | float | **Target**: Next month real price | 1.2660 |

### Data Statistics (Barley Example)

```
Price Range: 1.10 - 1.58 TRY/ton (real)
USD/TRY Range: 5.70 - 42.27 (7.4x increase!)
Volume: Highly variable (CV = 1.09)
```

---

## Yield Data Features (Y-band)

### File Format
- **Filename Pattern**: `features_Y-{crop}.csv`
- **Delimiter**: Comma (`,`)
- **Encoding**: UTF-8
- **Missing Values**: None

### Feature Categories

#### 1. Agricultural Metrics (4 features)
| Feature | Description | Unit |
|---------|-------------|------|
| `year` | Calendar year | YYYY |
| `harvest_area_ha` | Harvested area | hectares |
| `plant_area_ha` | Planted area | hectares |
| `production_mass_t` | Total production | tons |

#### 2. Temperature (36 features)
- `t2m_min_1` to `t2m_min_12`: Monthly minimum temperature (°C)
- `t2m_max_1` to `t2m_max_12`: Monthly maximum temperature (°C)
- `t2m_mean_1` to `t2m_mean_12`: Monthly mean temperature (°C)

#### 3. Precipitation (12 features)
- `precip_mm_1` to `precip_mm_12`: Monthly precipitation (millimeters)

#### 4. Extreme Weather Events (72 features)

**Heat Waves (24 features)**:
- `heatwave_35_1` to `heatwave_35_12`: Days above 35°C
- `heatwave_30_1` to `heatwave_30_12`: Days above 30°C

**Cold Events (12 features)**:
- `frost_1` to `frost_12`: Number of frost days (T < 0°C)

**Precipitation Extremes (24 features)**:
- `heavy_rain_1` to `heavy_rain_12`: Heavy rainfall days
- `dry_spell_max_1` to `dry_spell_max_12`: Maximum consecutive dry days

**Flood Risk (12 features)**:
- `flood_risk_1` to `flood_risk_12`: Days with flood risk

#### 5. Target (1 feature)
- `target_yield_t_ha`: **Target**: Yield in tons per hectare

### Data Statistics (Barley Example)

```
Yield Range: 1.92 - 3.57 t/ha
Average Temperature: 10.8 - 13.5°C (yearly average)
Average Precipitation: 19.5 - 33.4 mm (monthly average)
```

---

## Temporal Overlap

### Integration Window
The overlap period where both price and yield data are available:

```
Overlap Years: 2019, 2020, 2021, 2022, 2023, 2024 (6 years)
Monthly Samples: 64 months
  - 2019: 4 months (Sep-Dec)
  - 2020: 12 months
  - 2021: 12 months
  - 2022: 12 months
  - 2023: 12 months
  - 2024: 12 months
```

This overlap enables the integration model to learn the relationship between price and yield.

---

## Data Quality Assessment

### ✅ Strengths
1. **No Missing Values**: 100% complete data
2. **Consistent Format**: Standardized across all crops
3. **Rich Climate Data**: 13 different climate variables × 12 months
4. **Temporal Coverage**: Sufficient for time series modeling

---

## Data Preprocessing

### Required Transformations

#### 1. Scaling
```python
from sklearn.preprocessing import StandardScaler

# Price features (different scales)
scaler_price = StandardScaler()
X_price_scaled = scaler_price.fit_transform(X_price)

# Yield features (climate variables)
scaler_yield = StandardScaler()
X_yield_scaled = scaler_yield.fit_transform(X_yield)
```

## Data Sources

### Price Data
- **Source**: Turkish Grain Board (TMO - Toprak Mahsulleri Ofisi)
- **Type**: Exchange market data
- **Update Frequency**: Monthly
- **Currency**: Turkish Lira (TRY)
- **Inflation Adjustment**: Real prices calculated using CPI

### Yield & Climate Data
- **Agricultural Data**: TÜİK (Turkish Statistical Institute)
- **Climate Data**: ERA5 Climate Reanalysis
  - Provider: ECMWF (European Centre for Medium-Range Weather Forecasts)
  - Spatial Resolution: 0.25° × 0.25° (~30 km)
  - Temporal Resolution: Hourly data aggregated to monthly
  - Variables: Temperature, precipitation, extreme events

---

## Loading Data

### Basic Loading

```python
import pandas as pd

# Load price data
df_price = pd.read_csv('data/features_P-barley-monthly.csv')
df_price['date'] = pd.to_datetime(df_price['date'])

# Load yield data
df_yield = pd.read_csv('data/features_Y-barley.csv')

print(f"Price data shape: {df_price.shape}")
print(f"Yield data shape: {df_yield.shape}")
```

### Advanced Loading with Validation

```python
def load_and_validate_data(crop):
    """
    Load and validate data for a specific crop
    """
    # Load files
    price_file = f'data/features_P-{crop}-monthly.csv'
    yield_file = f'data/features_Y-{crop}.csv'
    
    df_price = pd.read_csv(price_file)
    df_yield = pd.read_csv(yield_file)
    
    # Validate
    assert df_price.isnull().sum().sum() == 0, "Missing values in price data!"
    assert df_yield.isnull().sum().sum() == 0, "Missing values in yield data!"
    
    assert len(df_price) == 75, f"Expected 75 price samples, got {len(df_price)}"
    assert len(df_yield) == 21, f"Expected 21 yield samples, got {len(df_yield)}"
    
    print(f"✅ {crop.title()} data loaded successfully!")
    print(f"   Price: {len(df_price)} months")
    print(f"   Yield: {len(df_yield)} years")
    
    return df_price, df_yield

# Usage
df_price_barley, df_yield_barley = load_and_validate_data('barley')
```

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{turkish_agricultural_forecast_2025,
  title={Turkish Grain Price and Yield Dataset},
  author={[Your Name]},
  year={2025},
  institution={Sabancı University},
  note={Compiled from TMO, TÜİK, and ERA5 sources}
}
```

---

*Last Updated: November 2025*
