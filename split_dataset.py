import os
import shutil
import random

# ===== CONFIG =====
INPUT_DIR = "dataset/personal"
OUTPUT_DIR = "dataset"
SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42

random.seed(SEED)

def split_dataset():
    for split in ["train", "val"]:
        for label in os.listdir(INPUT_DIR):
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

    for label in os.listdir(INPUT_DIR):
        label_path = os.path.join(INPUT_DIR, label)

        if not os.path.isdir(label_path):
            continue

        images = os.listdir(label_path)
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # ===== COPY TRAIN =====
        for img in train_imgs:
            src = os.path.join(label_path, img)
            dst = os.path.join(OUTPUT_DIR, "train", label, img)
            shutil.copy(src, dst)

        # ===== COPY VAL =====
        for img in val_imgs:
            src = os.path.join(label_path, img)
            dst = os.path.join(OUTPUT_DIR, "val", label, img)
            shutil.copy(src, dst)

        print(f"{label}: Train={len(train_imgs)}, Val={len(val_imgs)}")


if __name__ == "__main__":
    split_dataset()


# import os
# import random
# import shutil
# from pathlib import Path

# random.seed(42)

# src = Path("dataset/raw_combined")
# train_dir = Path("dataset/train")
# val_dir = Path("dataset/val")

# for label in os.listdir(src):
#     images = list((src / label).glob("*.*"))
#     random.shuffle(images)

#     split_idx = int(0.8 * len(images))

#     train_imgs = images[:split_idx]
#     val_imgs = images[split_idx:]

#     # create folders
#     (train_dir / label).mkdir(parents=True, exist_ok=True)
#     (val_dir / label).mkdir(parents=True, exist_ok=True)

#     # copy files
#     for img in train_imgs:
#         shutil.copy(img, train_dir / label / img.name)

#     for img in val_imgs:
#         shutil.copy(img, val_dir / label / img.name)

#     print(f"{label}: train={len(train_imgs)}, val={len(val_imgs)}")