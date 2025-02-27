import pandas as pd

# Đọc dữ liệu
station_day_path = "../data/station_day.csv"
df = pd.read_csv(station_day_path)

# Chuyển đổi cột Date sang kiểu datetime
df['Date'] = pd.to_datetime(df['Date'])

# Đặt Date làm index để xử lý dữ liệu theo thời gian
df.set_index('Date', inplace=True)

# Chuyển đổi object -> kiểu phù hợp trước khi nội suy (fix FutureWarning)
df = df.infer_objects(copy=False)

# Nội suy giá trị thiếu theo thời gian cho từng trạm, loại bỏ nhóm sau khi áp dụng (fix lỗi groupby)
df = df.groupby("StationId", group_keys=False).apply(lambda group: group.interpolate(method='time', limit_direction="both"))

# Xóa cột 'Xylene' vì có quá nhiều giá trị thiếu
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

# Reset index đúng cách (fix lỗi trùng cột 'StationId')
df.reset_index(drop=False, inplace=True)

# Lưu dữ liệu đã làm sạch vào file CSV mới
cleaned_file_path = "../clean_data/station_day_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Dữ liệu đã làm sạch và lưu vào: {cleaned_file_path}")
