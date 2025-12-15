import joblib

try:
    model = joblib.load("models/model.pkl")
    print("✅ Load thành công!")
    print(model)
except Exception as e:
    print("❌ Lỗi:", e)
