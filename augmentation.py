import os
import cv2
from pathlib import Path
import albumentations as A

# ===== CONFIG =====
TRAIN_DIR = "dataset/train"
AUG_PER_IMAGE = 2  # keep 2–3 initially

# ===== AUGMENTATION PIPELINE =====
class UsabilityAugmentationPipeline:
    def __init__(self):
        self.webcam_augmentations = A.Compose([
            A.ImageCompression(quality_lower=40, quality_upper=85, p=0.5),
            A.Downscale(scale_min=0.5, scale_max=0.8, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.4),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 40), p=0.3),
            A.CoarseDropout(
                max_holes=3, max_height=20, max_width=20,
                min_holes=1, min_height=10, min_width=10, p=0.2
            ),
        ])

    def augment(self, image):
        return self.webcam_augmentations(image=image)['image']


def augment_train():
    augmenter = UsabilityAugmentationPipeline()

    train_path = Path(TRAIN_DIR)

    for emotion_dir in train_path.iterdir():
        if not emotion_dir.is_dir():
            continue

        print(f"Processing class: {emotion_dir.name}")

        images = list(emotion_dir.glob("*.jpg"))

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            for i in range(AUG_PER_IMAGE):
                aug_img = augmenter.augment(img)

                new_name = f"{img_path.stem}_aug{i}.jpg"
                save_path = emotion_dir / new_name

                cv2.imwrite(str(save_path), aug_img)

    print("✅ Augmentation complete!")


if __name__ == "__main__":
    augment_train()