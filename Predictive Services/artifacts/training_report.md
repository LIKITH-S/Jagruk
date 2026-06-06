# Training Report – RandomForest

**Generated:** 2026-05-29 10:32 UTC

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.9917 |
| Precision | 1.0 |
| Recall | 0.6667 |
| F1 Score | 0.8 |
| Roc Auc | 1.0 |

## Classification Report

```
              precision    recall  f1-score   support

      Normal       0.99      1.00      1.00       352
      Hazard       1.00      0.67      0.80         9

    accuracy                           0.99       361
   macro avg       1.00      0.83      0.90       361
weighted avg       0.99      0.99      0.99       361

```

## Top Feature Importances

| Feature | Importance |
|---------|------------|
| temperature | 0.3049 |
| hotspot_density | 0.1188 |
| temperature_lag_1 | 0.1081 |
| temperature_roll_7d | 0.0997 |
| humidity_roll_7d | 0.0796 |
| humidity | 0.0650 |
| hotspot_density_lag_1d | 0.0477 |
| temperature_lag_24 | 0.0393 |
| soil_moisture_lag_1d | 0.0263 |
| soil_moisture | 0.0239 |
| rainfall_roll_7d | 0.0158 |
| soil_moisture_lag_3d | 0.0157 |
| hotspot_density_lag_3d | 0.0148 |
| elevation_risk_lag_1d | 0.0109 |
| humidity_trend | 0.0056 |