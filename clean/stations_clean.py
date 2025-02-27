import pandas as pd

# Đọc dữ liệu từ file
file_path = "../stations.csv" 
df = pd.read_csv(file_path)

# 📌 Kiểm tra thông tin dataset
print("📌 Các cột trong dataset:", df.columns.tolist())
print("🔍 Thông tin dữ liệu:")
print(df.info())

# 🔍 Kiểm tra giá trị thiếu
missing_values = df.isnull().sum()
print("🔍 Số lượng giá trị thiếu trên mỗi cột:\n", missing_values)

# ✅ Chuẩn hóa dữ liệu (xóa khoảng trắng, chuẩn hóa chữ cái đầu)
df['StationId'] = df['StationId'].astype(str).str.strip()
df['City'] = df['City'].astype(str).str.title()
df['State'] = df['State'].astype(str).str.title()

# ✅ Xử lý giá trị thiếu trong cột `Status` (nếu có)
if 'Status' in df.columns:
    df['Status'].fillna('Unknown', inplace=True)

# ✅ Loại bỏ dòng trùng lặp
df.drop_duplicates(inplace=True)

# ✅ Lưu dữ liệu đã làm sạch vào file mới
cleaned_file_path = "../clean_data/stations_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Dữ liệu đã làm sạch và lưu vào: {cleaned_file_path}")
