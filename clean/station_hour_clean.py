import pandas as pd

# Đọc dữ liệu
file_path = "../data/station_hour.csv"  # Cập nhật đường dẫn file của bạn
df = pd.read_csv(file_path)

# Chuyển đổi cột Date sang kiểu datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Đặt Datetime làm index để xử lý dữ liệu theo thời gian
df.set_index('Datetime', inplace=True)

# Chuyển đổi object -> kiểu phù hợp trước khi nội suy
df = df.infer_objects(copy=False)

# Nội suy giá trị thiếu theo thời gian cho từng trạm
df = df.groupby("StationId", group_keys=False).apply(lambda group: group.interpolate(method='time', limit_direction="both"))

# Xóa cột 'Xylene' nếu có quá nhiều giá trị thiếu
if 'Xylene' in df.columns:
    df.drop(columns=['Xylene'], inplace=True)

# Loại bỏ outliers bằng phương pháp IQR
def remove_outliers_iqr(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# Áp dụng loại bỏ outliers cho các cột số
num_cols = df.select_dtypes(include=['float64']).columns
df = remove_outliers_iqr(df, num_cols)

# Reset index sau khi xử lý
df.reset_index(drop=False, inplace=True)

# Lưu dữ liệu đã làm sạch vào file CSV mới
cleaned_file_path = "../clean_data/station_hour_cleaned.csv"  # Cập nhật đường dẫn lưu file
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Dữ liệu đã làm sạch và lưu vào: {cleaned_file_path}")
