import os
import random
from PIL import Image, ImageEnhance
import numpy as np

# =========================
# CẤU HÌNH
# =========================
DATA_DIR = "images_raw"      # thư mục chứa các lớp
TARGET_PER_CLASS = 50        # muốn mỗi lớp có tối thiểu bao nhiêu ảnh
IMG_SIZE = (224, 224)        # resize về cùng kích thước


# =========================
# HÀM TĂNG CƯỜNG ẢNH
# =========================
def random_augment(img: Image.Image) -> Image.Image:
    """Áp dụng 1 loạt biến đổi ngẫu nhiên lên ảnh."""
    # đảm bảo là RGB
    img = img.convert("RGB")

    # 1. Rotate nhẹ
    if random.random() < 0.7:
        angle = random.uniform(-25, 25)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # 2. Flip ngang
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 3. Flip dọc
    if random.random() < 0.3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 4. Brightness
    if random.random() < 0.7:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)

    # 5. Contrast
    if random.random() < 0.7:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)

    # 6. Color
    if random.random() < 0.5:
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)

    # 7. Thêm chút noise
    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 10, arr.shape)   # sigma = 10
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Resize cuối cùng cho chắc
    img = img.resize(IMG_SIZE)

    return img


# =========================
# XỬ LÝ TỪNG LỚP
# =========================
def augment_class(class_name: str):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        return

    # Ảnh gốc (không tính ảnh augment)
    base_images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not f.startswith("aug_")
    ]

    # Nếu lớp này đã đủ ảnh → bỏ qua
    current_count = len([
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if current_count >= TARGET_PER_CLASS:
        print(f"✅ Lớp {class_name} đã có {current_count} ảnh (>= {TARGET_PER_CLASS}), bỏ qua.")
        return

    if not base_images:
        print(f"⚠ Lớp {class_name} không có ảnh gốc, không augment được.")
        return

    print(f"📌 Lớp {class_name}: hiện có {current_count} ảnh, sẽ augment tới {TARGET_PER_CLASS} ảnh.")

    idx = 0
    while current_count < TARGET_PER_CLASS:
        idx += 1

        # Chọn ngẫu nhiên một ảnh gốc
        base_name = random.choice(base_images)
        base_path = os.path.join(class_dir, base_name)

        img = Image.open(base_path)
        aug_img = random_augment(img)

        # Đặt tên file mới
        new_name = f"aug_{current_count+1:04d}.jpg"
        save_path = os.path.join(class_dir, new_name)
        aug_img.save(save_path, quality=95)

        current_count += 1

    print(f"✅ Done lớp {class_name}: tổng cộng {current_count} ảnh.")


def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Không tìm thấy thư mục {DATA_DIR}")
        return

    classes = sorted([
        c for c in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, c))
    ])

    print("🔄 Bắt đầu augmentation cho các lớp:", classes)

    for cls in classes:
        augment_class(cls)

    print("🎉 Hoàn thành augmentation!")


if __name__ == "__main__":
    main()
