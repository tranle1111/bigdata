import pandas as pd
import numpy as np

# Đọc dữ liệu
city_day_path = "../data/city_day.csv"
df = pd.read_csv(city_day_path)

# Chuyển đổi kiểu dữ liệu
df['Date'] = pd.to_datetime(df['Date'])

# Đặt Date làm index để nội suy theo thời gian
df.set_index('Date', inplace=True)

# Chuyển các cột về dạng số nếu có kiểu object
df = df.infer_objects(copy=False)

# Xử lý giá trị thiếu - Nội suy theo thời gian
df.interpolate(method='time', inplace=True)

# Reset lại index sau khi nội suy
df.reset_index(inplace=True)

# Lưu dữ liệu đã xử lý
df.to_csv("../clean_data/city_day_cleaned.csv", index=False)

print("✅ Dữ liệu đã được làm sạch và lưu lại thành city_day_cleaned.csv")
