# Ex.No: 6               HOLT WINTERS METHOD
### Date: 26-05-2026



### AIM:
To implement the Holt-Winters forecasting method in Python for analyzing and forecasting time series data with trend and seasonality.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(
    "wb_commodity_price_intelligence_1960_2026.csv",
    parse_dates=['date'],
    index_col='date',
    engine='python' # Added engine='python' to handle potential parsing issues
)

# Display available commodities
print("Unique commodity names:")
print(data['commodity_name'].unique())

# Select commodity name
commodity = "Aluminum"   # Change this to any commodity you want

# Filter selected commodity data
commodity_data = data[data['commodity_name'] == commodity]

# Resample monthly price data
data_monthly_price = commodity_data['price_nominal_usd'].resample('MS').mean()

# Display first rows
print(data_monthly_price.head())

# Plot original prices
plt.figure(figsize=(12,6))
data_monthly_price.plot()

plt.title(f"Monthly {commodity} Prices")
plt.xlabel("Date")
plt.ylabel(f"{commodity} Price (USD)")
plt.grid(True)

plt.show()

# Scale data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data_monthly_price.values.reshape(-1,1)
    ).flatten(),
    index=data_monthly_price.index
)

# Plot scaled data
plt.figure(figsize=(12,6))
scaled_data.plot()

plt.title(f"Scaled {commodity} Prices")
plt.xlabel("Date")
plt.ylabel("Scaled Price")
plt.grid(True)

plt.show()

# Seasonal decomposition
decomposition = seasonal_decompose(
    data_monthly_price,
    model='additive',
    period=12
)

decomposition.plot()
plt.show()

# Shift scaled data for multiplicative model
scaled_data = scaled_data + 1

# Train-test split
train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Build Holt-Winters model
model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast on test data
test_predictions = model.forecast(steps=len(test_data))

# Plot train/test/predictions
fig, ax = plt.subplots(figsize=(12,6))

train_data.plot(ax=ax)
test_predictions.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend([
    'Training Data',
    f'Predicted {commodity} Prices',
    f'Actual {commodity} Prices'
])

ax.set_title(f'{commodity} Price Prediction Performance')
ax.set_xlabel('Date')
ax.set_ylabel(f'Scaled {commodity} Price')

plt.grid(True)
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))

print("RMSE Value:", rmse)

# Final model on full dataset
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast future values
future_steps = int(len(data_monthly_price) / 4)

final_predictions = final_model.forecast(steps=future_steps)

# Plot final forecast
fig, ax = plt.subplots(figsize=(12,6))

data_monthly_price.plot(ax=ax)
final_predictions.plot(ax=ax)

ax.legend([
    f'Actual {commodity} Prices',
    f'Forecasted {commodity} Prices'
])

ax.set_xlabel('Date')
ax.set_ylabel(f'{commodity} Price (USD)')

ax.set_title(
    f'{commodity} Price Forecast Using Holt-Winters Exponential Smoothing'
)

plt.grid(True)
plt.show()
```
### OUTPUT:
## TEST_PREDICTION:
<img width="767" height="414" alt="image" src="https://github.com/user-attachments/assets/eed3fef1-c321-451b-afe5-63476e56337d" />


<img width="722" height="526" alt="image" src="https://github.com/user-attachments/assets/ee47a951-b03b-42e8-8196-f1de64c7f09e" />


## FINAL_PREDICTION:
<img width="774" height="418" alt="image" src="https://github.com/user-attachments/assets/19d07cb9-a48e-4860-8c38-0d82ee77ab03" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
