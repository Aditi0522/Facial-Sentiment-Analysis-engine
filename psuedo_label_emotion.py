import os
import cv2
import shutil
import logging
from pathlib import Path

from emotiefflib.facial_analysis import EmotiEffLibRecognizer

# ================= CONFIG =================
INPUT_DIR = "dataset/personal_raw_faces"
OUTPUT_DIR = "dataset/personal"
CONF_THRESHOLD = 0.55

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PseudoLabel")

# ================= LABELS =================
VALID_LABELS = [
    'Angry',
    'Disgusted',
    'Fearful',
    'Happy',
    'Neutral',
    'Sad',
    'Surprised'
]

LABEL_MAP = {
    "Anger": "Angry",
    "Disgust": "Disgusted",
    "Fear": "Fearful",
    "Happiness": "Happy",
    "Neutral": "Neutral",
    "Sadness": "Sad",
    "Surprise": "Surprised"
}

# ================= INIT MODEL =================
model = EmotiEffLibRecognizer(
    engine="onnx",
    model_name="enet_b0_8_best_afew",
    device="cpu"   # change to "cuda" if GPU works
)

# ================= MAIN FUNCTION =================
def pseudo_label():
    # create output folders
    for label in VALID_LABELS:
        os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

    files = os.listdir(INPUT_DIR)
    logger.info(f"Found {len(files)} images")

    total = 0
    kept = 0

    for img_name in files:
        img_path = os.path.join(INPUT_DIR, img_name)

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read: {img_name}")
            continue

        try:
            # 🔥 Use cropped face directly (NO face detection again)
            emotion, scores = model.predict_emotions(img, logits=False)

            if not emotion:
                logger.warning(f"No prediction for {img_name}")
                continue

            raw_label = emotion[0]
            label = LABEL_MAP.get(raw_label, None)

            if label is None:
                continue
            confidence = float(max(scores[0]))

            print(label,confidence)

        except Exception as e:
            logger.error(f"Error processing {img_name}: {e}")
            continue

        total += 1

        # 🔥 confidence filtering
        if confidence < CONF_THRESHOLD:
            continue

        if label not in VALID_LABELS:
            continue

        save_path = os.path.join(
            OUTPUT_DIR,
            label,
            f"pseudo_{confidence:.2f}_{img_name}"
        )

        shutil.copy(img_path, save_path)
        kept += 1

        if kept % 100 == 0:
            logger.info(f"Saved {kept} images")

    # ===== SAFE LOGGING =====
    logger.info(f"Total processed: {total}")
    logger.info(f"Kept: {kept}")

    if total == 0:
        logger.warning("No valid predictions made")
    else:
        logger.info(f"Retention: {kept / total:.2f}")


# ================= RUN =================
if __name__ == "__main__":
    pseudo_label()