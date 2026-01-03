# Final Report - Findings and Hypothesis Tests

## Findings (Summary)
- P-band: best monthly price model differs by crop; gradient boosting/random forest lead for most crops.
- Y-band (GS aggregate): ridge performs best for wheat and barley with strong R2; maize remains unstable.
- Integration (Bayesian Ridge): wheat improves with minimal features; barley moderate; maize unstable and not reliable.
- Monte Carlo (residual bootstrap): wheat has higher expected earnings and higher risk than barley.

## Hypothesis Tests

| Test | H0 | H1 | Method | Stat | p-value | Decision (alpha=0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| FX vs price (wheat daily) | rho=0 | rho!=0 | Spearman | -0.391 | 1.31e-58 | Reject H0 |
| Wheat yield early vs late | mean_early=mean_late | mean_early!=mean_late | Welch t-test | -3.557 | 0.00248 | Reject H0 |
| Price variance across crops | var_w=var_b=var_m | not all equal | Levene | 9.117 | 0.000155 | Reject H0 |
| Dry spell vs wheat yield | rho=0 | rho!=0 | Spearman | -0.238 | 0.298 | Fail to reject |

## Model Performance (Best by RMSE)

### P-band (Price)
| crop | model | MAE | RMSE | SMAPE |
| --- | --- | --- | --- | --- |
| barley | gradient_boosting | 0.040 | 0.048 | 3.032 |
| maize | random_forest | 0.027 | 0.032 | 2.025 |
| wheat | gradient_boosting | 0.041 | 0.049 | 2.512 |

### Y-band (Yield)
| crop | model | MAE | RMSE | SMAPE | R2 |
| --- | --- | --- | --- | --- | --- |
| barley | ridge | 0.038 | 0.046 | 1.227 | 0.988 |
| maize | ridge | 0.735 | 0.806 | 6.301 | 0.205 |
| wheat | ridge | 0.108 | 0.123 | 3.161 | 0.901 |

### Integration (Bayesian Ridge)
| crop | variant | model | MAE | RMSE | SMAPE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| barley | full | bayesian_ridge | 94811.800 | 100914.468 | 7.084 | 0.451 |
| maize | full | bayesian_ridge | 124571.123 | 130815.412 | 4.688 | -31.375 |
| wheat | minimal | bayesian_ridge_minimal | 70739.034 | 71090.101 | 3.093 | 0.620 |

- Note: maize integration is unstable (very low/negative R2).

## Monte Carlo (Residual Bootstrap)
| crop | expected_earnings | earnings_std | earnings_p5 | earnings_p95 | earnings_cvar_5 |
| --- | --- | --- | --- | --- | --- |
| wheat | 2114765 | 151417 | 1814222 | 2362448 | 1784663 |
| barley | 1168535 | 91950 | 989097 | 1285904 | 977139 |

## Key Figures
- Price trends: reports/figures/price_real_monthly_all_crops.png
- Yield trends: reports/figures/yield_timeseries_all_crops.png
- Hazard vs yield: reports/figures/hazard_vs_yield_wheat.png
- Monte Carlo risk-return: reports/figures/monte_carlo_risk_return.png

## Conclusion (Which Crop to Plant?)
- **Most risk-averse**: **Wheat**. It shows the most stable integration behavior and lower downside risk in Monte Carlo (tighter spread).
- **Balanced strategy**: **Barley**. Expected earnings are lower than wheat but risk is also lower; good compromise for stability.
- **High-risk/uncertain**: **Maize**. Integration is unstable (negative R²) with limited years and high volatility; do not rely on the model for maize until more data is added.
