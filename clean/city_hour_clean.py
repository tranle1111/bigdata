import pandas as pd

# Đọc dữ liệu
city_hour_path = "../data/city_hour.csv"
df = pd.read_csv(city_hour_path)

# Chuyển đổi cột 'Datetime' sang kiểu datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Nội suy giá trị thiếu theo thời gian cho từng thành phố
df = df.groupby("City", group_keys=False).apply(lambda group: 
    group.set_index('Datetime').interpolate(method='time', limit_direction="both").reset_index())

# Hàm loại bỏ outliers bằng IQR
def remove_outliers_iqr(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# Áp dụng loại bỏ outliers cho các cột ô nhiễm
pollutant_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'AQI']
df = remove_outliers_iqr(df, pollutant_columns)

# Xóa các cột có quá nhiều giá trị thiếu (trên 40% dữ liệu thiếu)
missing_threshold = 0.4  # Ngưỡng 40%
df = df.dropna(axis=1, thresh=int((1 - missing_threshold) * len(df)))

# Sắp xếp lại dữ liệu theo 'Datetime'
df = df.sort_values(by='Datetime')

# Lưu dữ liệu đã xử lý
cleaned_file_path = "../clean_data/city_hour_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Dữ liệu đã làm sạch và lưu vào {cleaned_file_path}")
