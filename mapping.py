
# import os

# base = "dataset/raw_combined"

# for label in os.listdir(base):
#     count = len(os.listdir(os.path.join(base, label)))
#     print(label, count)


# import shutil
# from pathlib import Path

# AFFECTNET_FOLDER_MAP = {
#     "anger": "Angry",
#     "disgust": "Disgusted",
#     "fear": "Fearful",
#     "happy": "Happy",
#     "neutral": "Neutral",
#     "sad": "Sad",
#     "surprise": "Surprised"
# }

# src_base = Path("raw_dataset/affectnet/archive (3)/Train")
# dst_base = Path("dataset/raw_combined")

# # create output folders
# for label in AFFECTNET_FOLDER_MAP.values():
#     (dst_base / label).mkdir(parents=True, exist_ok=True)

# # limit per class (VERY IMPORTANT)
# MAX_PER_CLASS = 2000

# for src_folder, target_label in AFFECTNET_FOLDER_MAP.items():
#     src_path = src_base / src_folder
#     dst_path = dst_base / target_label

#     count = 0

#     for img_path in src_path.glob("*.*"):
#         if count >= MAX_PER_CLASS:
#             break

#         new_name = f"affect_{src_folder}_{count}.jpg"
#         shutil.copy(img_path, dst_path / new_name)

#         count += 1

#     print(f"{target_label}: {count} images copied")

# print("Done ✅")


# import pandas as pd
# import shutil
# from pathlib import Path

# RAF_MAP = {
#     1: 'Surprised',
#     2: 'Fearful',
#     3: 'Disgusted',
#     4: 'Happy',
#     5: 'Sad',
#     6: 'Angry',
#     7: 'Neutral'
# }


# base = Path("raw_dataset/raf-db")
# data_dir = base / "DATASET"
# out = Path("dataset/raw_combined")

# df = pd.read_csv(base / "train_labels.csv")

# for _, row in df.iterrows():
#     img_name = row["image"]
#     label_id = int(row["label"])

#     label = RAF_MAP[label_id]

#     img_path = data_dir / "train" / str(label_id) / img_name

#     if img_path.exists():
#         dst_path = out / label / f"raf_{img_name}"
#         shutil.copy(img_path, dst_path)

import os
import shutil
import random
from pathlib import Path

# ===== CONFIG =====
FER_DIR = "raw_dataset/fer+/train"
DATASET_DIR = "dataset"
SPLIT_RATIO = 0.8
SEED = 42

random.seed(SEED)

# ===== LABEL MAP =====
LABEL_MAP = {
    "suprise": "Surprised"
}

def merge_ferplus():
    fer_path = Path(FER_DIR)

    for fer_label_dir in fer_path.iterdir():
        if not fer_label_dir.is_dir():
            continue

        fer_label = fer_label_dir.name.lower()
        mapped_label = LABEL_MAP.get(fer_label)

        if mapped_label is None:
            print(f"Skipping {fer_label}")
            continue

        images = list(fer_label_dir.glob("*.png"))
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # ===== COPY TRAIN =====
        for img_path in train_imgs:
            dst = Path(DATASET_DIR) / "train" / mapped_label / img_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dst)

        # ===== COPY VAL =====
        for img_path in val_imgs:
            dst = Path(DATASET_DIR) / "val" / mapped_label / img_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dst)

        print(f"{mapped_label}: Train={len(train_imgs)}, Val={len(val_imgs)}")


if __name__ == "__main__":
    merge_ferplus()