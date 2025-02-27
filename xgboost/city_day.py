import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import joblib

# Đọc dữ liệu
file_path = "../clean_data/city_day_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])

# Chọn các cột đầu vào và mục tiêu
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
target = 'AQI'
df = df.dropna(subset=features + [target])

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Huấn luyện XGBoost Regression
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)

# Đánh giá mô hình XGBoost
print(f"🎯 [XGBoost] RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"🎯 [XGBoost] MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"🎯 [XGBoost] R² Score: {r2_score(y_test, y_pred)}")

# === TIME SERIES FORECASTING (ARIMA) ===
df.set_index('Date', inplace=True)
df_numeric = df.select_dtypes(include=[np.number])  # Chỉ lấy cột số
df_resampled = df_numeric.resample('D').mean()  # Resample theo ngày

df_resampled.dropna(inplace=True)  # Xóa các giá trị NaN sau resample

train_size = int(len(df_resampled) * 0.8)
train, test = df_resampled.iloc[:train_size], df_resampled.iloc[train_size:]

# Huấn luyện mô hình ARIMA
model_arima = ARIMA(train['AQI'], order=(5,1,0))
model_fit = model_arima.fit()
forecast = model_fit.forecast(steps=len(test))

# Vẽ biểu đồ dự báo
plt.figure(figsize=(10,5))
plt.plot(train.index, train['AQI'], label='Train')
plt.plot(test.index, test['AQI'], label='Actual')
plt.plot(test.index, forecast, label='Forecast', linestyle='dashed')
plt.legend()
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('AQI Forecasting using ARIMA')
plt.show()

joblib.dump(model_xgb, "../Mo_hinh_huan_luyen/xgboost_aqi_model_city_day.pkl")
print("✅ Mô hình XGBoost đã được lưu thành công!")