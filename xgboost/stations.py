import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu
file_path = "../clean_data/stations_cleaned.csv"
df = pd.read_csv(file_path)

# Chọn các cột cần thiết
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
target = 'AQI'

# Loại bỏ dòng có giá trị NaN trong các cột quan trọng
df_cleaned = df.dropna(subset=features + [target])

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    df_cleaned[features], df_cleaned[target], test_size=0.2, random_state=42
)

# Huấn luyện XGBoost Regression
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model_xgb.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model_xgb.predict(X_test)

# Đánh giá mô hình XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
mae_xgb = mean_absolute_error(y_test, y_pred)
r2_xgb = r2_score(y_test, y_pred)

print(f"XGBoost - RMSE: {rmse_xgb}, MAE: {mae_xgb}, R2: {r2_xgb}")
