import os
import numpy as np
from PIL import Image
import autokeras as ak
import joblib
from sklearn.model_selection import train_test_split


# ============================
# CONFIG
# ============================
DATA_DIR = "images_raw"
IMG_SIZE = (224, 224)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "waste_model")   # SavedModel export
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")


# ============================
# LOAD DATASET
# ============================
def load_dataset():
    X, y = [], []

    classes = sorted([
        c for c in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, c))
    ])

    print(f"📌 Các lớp tìm thấy: {classes}")

    for label in classes:
        folder = os.path.join(DATA_DIR, label)

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img = Image.open(os.path.join(folder, file)).convert("RGB")
                img = img.resize(IMG_SIZE)

                X.append(np.array(img))
                y.append(label)

    print(f"📂 Tổng ảnh load được: {len(X)}")

    return np.array(X), np.array(y), classes


# ============================
# AUTO-KERAS TRAINING
# ============================
def train_automl():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("🔄 Loading dataset...")
    X, y, classes = load_dataset()

    # Save labels for prediction use
    joblib.dump(classes, LABELS_PATH)
    print("💾 labels.pkl đã được lưu.")

    # Check stratify condition
    class_counts = {cls: np.sum(y == cls) for cls in classes}
    print("📊 Số lượng mỗi lớp:", class_counts)

    if min(class_counts.values()) < 2:
        print("⚠ Lớp nào đó có < 2 ảnh → Không dùng stratify.")
        stratify_val = None
    else:
        stratify_val = y

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_val
    )

    print("🚀 Training AutoKeras model...")
    clf = ak.ImageClassifier(
        overwrite=True,
        max_trials=10,
        project_name="waste_classifier"
    )

    clf.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20
    )

    print("🎉 Training completed!")

    # EXPORT MODEL — THE ONLY WAY ON KERAS 3.x
    print("💾 Exporting SavedModel (Keras 3 compatible)...")
    model = clf.export_model()

    # ✔ Đây là lệnh đúng nhất để tạo SavedModel
    model.export(MODEL_PATH)

    print(f"✅ Model đã export SavedModel tại: {MODEL_PATH}")
    print(f"📦 Labels đã lưu: {LABELS_PATH}")
    print("🔥 DONE – READY FOR STREAMLIT!")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    train_automl()
