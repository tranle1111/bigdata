from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import numpy as np
import joblib

app = Flask(__name__)

# Load mô hình đã huấn luyện
model_xgb = joblib.load("../Mo_hinh_huan_luyen/xgboost_aqi_model_city_day.pkl")

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']

# Hàm đánh giá chất lượng không khí
def get_air_quality(aqi):
    if aqi <= 50:
        return "Chat luong khong khi tot"
    elif aqi <= 100:
        return "Chat luong khong khi trung binh"
    elif aqi <= 150:
        return "Chat luong khong khi kem"
    elif aqi <= 200:
        return "Chat luong khong khi rat kem"
    elif aqi <= 300:
        return "Chat luong khong khi nguy hai"
    else:
        return "Chat luong khong khi nguy hiem"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Kiểm tra xem tất cả các cột có tồn tại không
        for col in features:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        # Dự đoán AQI
        prediction = model_xgb.predict(df[features])
        aqi_value = float(prediction[0])
        air_quality = get_air_quality(aqi_value)
        
        return jsonify({
            "predicted_AQI": aqi_value,
            "air_quality": air_quality
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
