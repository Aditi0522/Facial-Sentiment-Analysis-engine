import torch
import torch.nn.functional as F
import cv2
import numpy as np
from temporal_smoothing import TemporalSmoother, EmotionTimeline, aggregate_summary

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

from timm import create_model
import torch.nn as nn

class EmotiEffLibFineTuned(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ─────────────────────────────────────────────────────────────
# PREPROCESS FUNCTION (IMPORTANT)
# ─────────────────────────────────────────────────────────────
def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std

    frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
    return torch.tensor(frame, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION (THIS IS WHAT YOU CALL)
# ─────────────────────────────────────────────────────────────
def run_temporal_emotion_analysis(video_path, model_path):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")

    # Load model
    model = EmotiEffLibFineTuned(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(" Model loaded")

    # Initialize temporal system
    smoother = TemporalSmoother(window_size=5)
    timeline = EmotionTimeline(smoother)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    processed = 0

    print(" Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: process every 3rd frame (faster)
        if frame_idx % 3 != 0:
            frame_idx += 1
            continue

        timestamp_ms = int((frame_idx / fps) * 1000)

        # Preprocess
        img = preprocess(frame).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = model(img)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        # Convert to dict
        raw_emotions = {EMOTIONS[i]: float(probs[i]) for i in range(7)}
        confidence = float(np.max(probs))

        # Add to temporal pipeline
        timeline.add_frame(timestamp_ms, raw_emotions, confidence)

        processed += 1
        frame_idx += 1

    cap.release()

    print(f" Frames processed: {processed}")

    # Build outputs
    segments = timeline.build_segments(min_segment_ms=500)
    summary = aggregate_summary(timeline)

    print("\n SEGMENTS:")
    for s in segments:
        print(s)

    print("\n SUMMARY:")
    print(summary)

    return {
        "segments": segments,
        "summary": summary
    }


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_temporal_emotion_analysis(
        video_path=".\self_vid_data_raw\WIN_20260323_22_48_44_Pro.mp4",        # change this
        model_path="./fine-tuned-model/best_model.pt"          # your trained model
    )