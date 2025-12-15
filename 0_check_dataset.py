import os

dataset_path = "images_raw/"

classes = os.listdir(dataset_path)

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{cls:10s} : {count} images")
