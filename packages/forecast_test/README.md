# forecast_testing

## Objective

The `forecast_testing` package is designed to simplify the process of time series forecasting for multiple related metrics. It's particularly useful for businesses that need to predict future values of interconnected metrics, such as installs and trials for a product.

Key features:
- Automated feature engineering for time series data
- Multiple model comparison (Neural Network, Linear Regression, Random Forest)
- Hierarchical prediction for related metrics
- Easy-to-use interface for quick forecasting

This package is ideal for data scientists and analysts who want to quickly test and implement forecasting models without the need for extensive custom code.

## Functions

### 1. predict_future

The main function of the package. It handles the entire prediction process from data preparation to final forecasting.

```
from forecast_testing import predict_future
result = predict_future(df, 'Date', ('1/1/2022', '1/23/2025'), ('1/24/2025', '3/1/2025'), ['installs', 'trials'])
```


Parameters:
- `df`: pandas DataFrame containing your time series data
- `date_col`: string, name of the date column in your DataFrame
- `train_range`: tuple of strings (start_date, end_date) for the training period
- `pred_range`: tuple of strings (start_date, end_date) for the prediction period
- `targets`: list of strings, names of the target columns to predict

Returns:
- pandas DataFrame with predictions for the specified targets over the prediction period

### 2. create_features

A utility function for feature engineering on time series data.

```
from forecast_testing import create_features
df_with_features = create_features(df, 'Date')
```

Parameters:
- `df`: pandas DataFrame containing your time series data
- `date_col`: string, name of the date column in your DataFrame

Returns:
- pandas DataFrame with additional time-based features

### 3. PredictionModel

A class that encapsulates the model training and selection process.

```
from forecast_testing import PredictionModel
model = PredictionModel(features)
model.train_models(X, y)
model.select_best_model(X, y)
predictions = model.predict(X_new)
```

This class is mainly used internally by `predict_future`, but advanced users can access it for more control over the modeling process.

## Future Scope

1. Model Customization: Allow users to specify custom models or hyperparameters.
2. Feature Selection: Implement automated feature selection techniques to improve model performance.
3. Seasonality Handling: Add explicit handling of seasonal patterns in the data.
4. Uncertainty Quantification: Implement prediction intervals or other measures of forecast uncertainty.
5. Multi-step Forecasting: Extend the package to handle multi-step ahead forecasts more robustly.
6. External Regressors: Allow users to specify additional external variables that might influence the forecasts.
7. Model Interpretability: Add functions to explain model predictions and feature importance.
8. Cross-validation: Implement time series cross-validation for more robust model evaluation.
9. Anomaly Detection: Add capabilities to detect and handle anomalies in the input data.
10. Visualization: Include functions for automatic plotting of forecasts and model diagnostics.

By addressing these areas, the `forecast_testing` package could become a more comprehensive and powerful tool for time series forecasting across a wide range of applications.