"""
═══════════════════════════════════════════════════════════════════════
TEMPORAL SMOOTHING + EMOTION TIMELINE + USABILITY METRICS SCRIPT
═══════════════════════════════════════════════════════════════════════

This script processes a usability test video end-to-end:
  SCRFD (face detection) → EmotiEffLib (emotion recognition) →
  Temporal Smoothing → Emotion Timeline → Usability Metrics →
  Summary Aggregation → JSON output

HOW TO RUN:
    python temporal_smoothing_pipeline.py

BEFORE RUNNING — change these paths in the CONFIG section below:
    1. MODEL_PATH    → path to your fine-tuned best_model.pt
    2. VIDEO_PATH    → path to the video file to analyze
    3. OUTPUT_DIR    → where to save results (JSON + timeline chart)

REQUIRED PACKAGES:
    pip install torch torchvision timm opencv-python insightface onnxruntime numpy matplotlib

OUTPUT FILES (saved to OUTPUT_DIR):
    • emotion_results.json       — full timeline, segments, summary, UX metrics
    • emotion_timeline_chart.png — visual chart of emotions over time
    • emotion_segments.txt       — human-readable segment list
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
from timm import create_model
from torchvision import transforms
from PIL import Image

# ═══════════════════════════════════════════════════════════════════
# ██  CONFIG — CHANGE THESE BEFORE RUNNING  ██
# ═══════════════════════════════════════════════════════════════════

# Path to your fine-tuned model checkpoint
MODEL_PATH = "./fine-tuned-model/best_model.pt"

# Path to the video file to analyze
VIDEO_PATH = "./self_vid_data_raw/WIN_20260323_22_48_44_Pro.mp4"

# Where to save output files
OUTPUT_DIR = "./temporal_results"

# How many frames per second to process (lower = faster, higher = more detail)
# 10 is a good default. Set to 5 for long videos, 15 for short clips.
PROCESS_FPS = 10

# Temporal smoothing window size (in frames, not seconds)
# With PROCESS_FPS=10, window=5 means 0.5-second smoothing window.
# Increase for smoother output, decrease for more responsive tracking.
SMOOTHING_WINDOW = 5

# Minimum segment duration in milliseconds.
# Segments shorter than this are merged into neighbors.
# 500ms is a good default. Increase to 1000 for cleaner timelines.
MIN_SEGMENT_MS = 500

# SCRFD face detection size. (640, 640) is default.
# Use (320, 320) for faster detection on low-res video.
DET_SIZE = (640, 640)

# ═══════════════════════════════════════════════════════════════════
# ██  DO NOT MODIFY BELOW THIS LINE  ██
# ═══════════════════════════════════════════════════════════════════

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']


# ── Model Definition (same as training) ─────────────────────────

class EmotiEffLibFineTuned(nn.Module):
    """EfficientNet-B0 backbone + 7-class emotion classifier."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = create_model('efficientnet_b0', pretrained=False, num_classes=0)
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── Image Preprocessing ─────────────────────────────────────────

# Same normalization used during training (ImageNet stats)
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_face(face_bgr):
    """
    Convert an OpenCV BGR face crop to a model-ready tensor.
    
    Args:
        face_bgr: numpy array (H, W, 3) in BGR color order (from OpenCV)
    Returns:
        tensor: (1, 3, 224, 224) normalized tensor ready for model
    """
    import cv2
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)
    tensor = PREPROCESS(pil_image)
    return tensor.unsqueeze(0)  # add batch dimension


# ── Temporal Smoother ────────────────────────────────────────────

class TemporalSmoother:
    """
    Confidence-weighted moving average filter.
    
    Removes frame-to-frame prediction flicker by averaging the last
    N predictions, weighted by each prediction's confidence.
    
    Example without smoothing (10 FPS):
        Frame 1: Happy (0.70)
        Frame 2: Neutral (0.50) ← noise
        Frame 3: Happy (0.65)
        Frame 4: Happy (0.72)
        Frame 5: Surprised (0.40) ← noise
    
    After smoothing (window=5):
        → Happy (0.61) — stable, correct prediction
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def smooth(self, emotions: dict, confidence: float) -> tuple:
        """
        Add a new frame's prediction to the buffer and return the
        smoothed (averaged) prediction.
        
        Args:
            emotions:   dict of 7-class probabilities from the model
            confidence: float, the max probability (dominant emotion score)
        
        Returns:
            (smoothed_emotions, smoothed_confidence)
        """
        self.buffer.append((emotions, confidence))

        smoothed = {e: 0.0 for e in EMOTIONS}
        conf_sum = 0.0

        for emo, conf in self.buffer:
            for emotion in EMOTIONS:
                smoothed[emotion] += emo.get(emotion, 0) * conf
            conf_sum += conf

        if conf_sum > 0:
            smoothed = {k: round(v / conf_sum, 4) for k, v in smoothed.items()}

        smoothed_confidence = round(max(smoothed.values()), 4)
        return smoothed, smoothed_confidence

    def reset(self):
        """Clear the buffer (call when starting a new video)."""
        self.buffer.clear()


# ── Emotion Timeline Generator ──────────────────────────────────

class EmotionTimeline:
    """
    Collects per-frame predictions and groups consecutive frames
    with the same dominant emotion into named segments.
    
    Output example:
        Segment 1:  0.0s – 3.2s  → Happy   (conf: 0.72)
        Segment 2:  3.2s – 5.1s  → Neutral  (conf: 0.61)
        Segment 3:  5.1s – 6.8s  → Angry    (conf: 0.58)
    """

    def __init__(self, smoother: TemporalSmoother):
        self.smoother = smoother
        self.raw_frames = []
        self.segments = []

    def add_frame(self, timestamp_ms: int, raw_emotions: dict, confidence: float):
        """Add one frame's raw prediction. Smoothing is applied internally."""
        smoothed_emotions, smoothed_confidence = self.smoother.smooth(
            raw_emotions, confidence
        )
        dominant = max(smoothed_emotions, key=smoothed_emotions.get)

        self.raw_frames.append({
            'timestamp_ms': timestamp_ms,
            'raw_emotions': raw_emotions,
            'smoothed_emotions': smoothed_emotions,
            'confidence': smoothed_confidence,
            'dominant_emotion': dominant,
        })

    def build_segments(self, min_segment_ms: int = 500) -> list:
        """
        Group consecutive same-emotion frames into segments.
        Merge micro-segments (< min_segment_ms) into neighbors.
        """
        if not self.raw_frames:
            return []

        segments = []
        cur = {
            'start_ms': self.raw_frames[0]['timestamp_ms'],
            'end_ms': self.raw_frames[0]['timestamp_ms'],
            'dominant_emotion': self.raw_frames[0]['dominant_emotion'],
            'avg_emotions': {e: [] for e in EMOTIONS},
            'avg_confidence': [],
        }

        for frame in self.raw_frames:
            if frame['dominant_emotion'] == cur['dominant_emotion']:
                cur['end_ms'] = frame['timestamp_ms']
                for e in EMOTIONS:
                    cur['avg_emotions'][e].append(frame['smoothed_emotions'].get(e, 0))
                cur['avg_confidence'].append(frame['confidence'])
            else:
                segments.append(self._finalize(cur))
                cur = {
                    'start_ms': frame['timestamp_ms'],
                    'end_ms': frame['timestamp_ms'],
                    'dominant_emotion': frame['dominant_emotion'],
                    'avg_emotions': {e: [frame['smoothed_emotions'].get(e, 0)] for e in EMOTIONS},
                    'avg_confidence': [frame['confidence']],
                }

        segments.append(self._finalize(cur))
        segments = self._merge_short(segments, min_segment_ms)
        self.segments = segments
        return segments

    def _finalize(self, seg):
        return {
            'start_ms': seg['start_ms'],
            'end_ms': seg['end_ms'],
            'duration_ms': seg['end_ms'] - seg['start_ms'],
            'dominant_emotion': seg['dominant_emotion'],
            'emotions': {e: round(float(np.mean(v)), 4) for e, v in seg['avg_emotions'].items()},
            'confidence': round(float(np.mean(seg['avg_confidence'])), 4),
        }

    def _merge_short(self, segments, min_ms):
        if len(segments) <= 1:
            return segments
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg['duration_ms'] < min_ms:
                merged[-1]['end_ms'] = seg['end_ms']
                merged[-1]['duration_ms'] = merged[-1]['end_ms'] - merged[-1]['start_ms']
            else:
                merged.append(seg)
        return merged


# ── Summary Aggregation (Confidence-Weighted) ────────────────────

def aggregate_summary(timeline: EmotionTimeline) -> dict:
    """
    Compute a single overall emotion summary from all frames.
    High-confidence frames contribute more than low-confidence frames.
    """
    frames = timeline.raw_frames
    if not frames:
        return {'error': 'No frames processed'}

    weighted = {e: 0.0 for e in EMOTIONS}
    total_w = 0.0

    for f in frames:
        w = f['confidence']
        for e in EMOTIONS:
            weighted[e] += f['smoothed_emotions'].get(e, 0) * w
        total_w += w

    if total_w > 0:
        overall = {e: round(v / total_w, 4) for e, v in weighted.items()}
    else:
        overall = {e: round(1 / 7, 4) for e in EMOTIONS}

    dominant = max(overall, key=overall.get)

    # Time distribution: what % of total time was each emotion dominant?
    total_time = frames[-1]['timestamp_ms'] - frames[0]['timestamp_ms']
    time_dist = {e: 0 for e in EMOTIONS}
    for seg in timeline.segments:
        time_dist[seg['dominant_emotion']] += seg['duration_ms']
    if total_time > 0:
        time_dist = {e: round(t / total_time, 3) for e, t in time_dist.items()}

    return {
        'overall_emotions': overall,
        'dominant_emotion': dominant,
        'dominant_confidence': overall[dominant],
        'time_distribution': time_dist,
        'total_frames': len(frames),
        'total_duration_ms': total_time,
        'avg_confidence': round(float(np.mean([f['confidence'] for f in frames])), 3),
    }


# ── Usability Metrics Calculator ────────────────────────────────

class UsabilityMetricsCalculator:
    """
    Derives UX-relevant scores from 7-class emotion probabilities.
    These are composite metrics, not new emotion classes.
    """

    def compute(self, emotions: dict, confidence: float) -> dict:
        a = emotions.get('Angry', 0)
        d = emotions.get('Disgusted', 0)
        f = emotions.get('Fearful', 0)
        h = emotions.get('Happy', 0)
        n = emotions.get('Neutral', 0)
        sa = emotions.get('Sad', 0)
        su = emotions.get('Surprised', 0)

        frustration = a * 0.5 + d * 0.3 + sa * 0.2
        confusion = f * 0.4 + su * 0.3 + (1 - confidence) * 0.3
        engagement = h * 0.3 + su * 0.3 + (1 - n) * 0.4
        satisfaction = h * 0.6 + n * 0.3 * confidence + su * 0.1
        emotional_intensity = 1 - n
        boredom = (n * 0.5 + sa * 0.3 + d * 0.2) * (1 - emotional_intensity)
        stress = f * 0.5 + a * 0.3 + su * 0.2

        return {
            'frustration_index': round(float(min(1.0, frustration)), 4),
            'confusion_score': round(float(min(1.0, confusion)), 4),
            'engagement_level': round(float(min(1.0, engagement)), 4),
            'satisfaction_score': round(float(min(1.0, satisfaction)), 4),
            'boredom_score': round(float(min(1.0, boredom)), 4),
            'stress_level': round(float(min(1.0, stress)), 4),
        }

    def detect_events(self, emotions: dict, confidence: float,
                      prev_emotions: dict = None) -> list:
        """Detect discrete usability events worth flagging."""
        events = []
        m = self.compute(emotions, confidence)

        if m['frustration_index'] > 0.6:
            events.append({
                'type': 'frustration_spike',
                'severity': 'high' if m['frustration_index'] > 0.8 else 'medium',
                'detail': f"frustration={m['frustration_index']:.2f}",
            })

        if m['confusion_score'] > 0.5:
            events.append({
                'type': 'confusion_detected',
                'severity': 'high' if m['confusion_score'] > 0.7 else 'medium',
                'detail': f"confusion={m['confusion_score']:.2f}",
            })

        if emotions.get('Happy', 0) > 0.8 and confidence > 0.7:
            events.append({
                'type': 'delight_moment',
                'severity': 'positive',
                'detail': f"happy={emotions['Happy']:.2f}, conf={confidence:.2f}",
            })

        if prev_emotions:
            prev_dom = max(prev_emotions, key=prev_emotions.get)
            curr_dom = max(emotions, key=emotions.get)
            if prev_dom != curr_dom:
                shift = abs(emotions[curr_dom] - prev_emotions.get(curr_dom, 0))
                if shift > 0.4:
                    events.append({
                        'type': 'emotion_shift',
                        'from': prev_dom,
                        'to': curr_dom,
                        'magnitude': round(float(shift), 2),
                    })

        return events


# ── Timeline Chart Generator ────────────────────────────────────

def save_timeline_chart(timeline: EmotionTimeline, output_path: str):
    """Save a visual chart showing emotion probabilities over time."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed. Skipping chart. pip install matplotlib")
        return

    frames = timeline.raw_frames
    if not frames:
        return

    timestamps = [f['timestamp_ms'] / 1000.0 for f in frames]  # convert to seconds
    emotion_data = {e: [f['smoothed_emotions'].get(e, 0) for f in frames] for e in EMOTIONS}

    colors = {
        'Angry': '#e74c3c', 'Disgusted': '#8e44ad', 'Fearful': '#e67e22',
        'Happy': '#2ecc71', 'Neutral': '#95a5a6', 'Sad': '#3498db',
        'Surprised': '#f1c40f',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    # Top: stacked area chart of emotion probabilities
    bottom = np.zeros(len(timestamps))
    for emotion in EMOTIONS:
        values = np.array(emotion_data[emotion])
        ax1.fill_between(timestamps, bottom, bottom + values,
                         alpha=0.7, color=colors[emotion], label=emotion)
        bottom += values

    ax1.set_ylabel('Emotion Probability', fontsize=12)
    ax1.set_title('Emotion Timeline (Temporally Smoothed)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(timestamps[0], timestamps[-1])
    ax1.set_ylim(0, 1)

    # Bottom: segment bars
    for seg in timeline.segments:
        start = seg['start_ms'] / 1000.0
        width = seg['duration_ms'] / 1000.0
        color = colors.get(seg['dominant_emotion'], '#999')
        ax2.barh(0, width, left=start, height=0.5, color=color, edgecolor='white')
        if width > 1.0:  # only label segments wide enough
            ax2.text(start + width / 2, 0, seg['dominant_emotion'],
                     ha='center', va='center', fontsize=8, fontweight='bold')

    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Segment', fontsize=10)
    ax2.set_yticks([])
    ax2.set_xlim(timestamps[0], timestamps[-1])

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Timeline chart saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# ██  MAIN PIPELINE  ██
# ═══════════════════════════════════════════════════════════════════

def main():
    import cv2

    print("=" * 65)
    print("  FACIAL EMOTION — TEMPORAL SMOOTHING PIPELINE")
    print("=" * 65)
    print(f"  Model:          {MODEL_PATH}")
    print(f"  Video:          {VIDEO_PATH}")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"  Process FPS:    {PROCESS_FPS}")
    print(f"  Smooth Window:  {SMOOTHING_WINDOW} frames")
    print(f"  Min Segment:    {MIN_SEGMENT_MS} ms")
    print()

    # ── Validate paths ──
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at '{MODEL_PATH}'")
        print(f"   → Update MODEL_PATH in the CONFIG section.")
        sys.exit(1)

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video not found at '{VIDEO_PATH}'")
        print(f"   → Update VIDEO_PATH in the CONFIG section.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Setup device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── Load emotion model ──
    print("  Loading EmotiEffLib model...")
    emotion_model = EmotiEffLibFineTuned(num_classes=7)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    emotion_model.load_state_dict(state_dict)
    emotion_model = emotion_model.to(device)
    emotion_model.eval()
    print("  ✓ Emotion model loaded.\n")

    # ── Load SCRFD face detector ──
    print("  Loading SCRFD face detector (buffalo_s)...")
    from insightface.app import FaceAnalysis
    face_detector = FaceAnalysis(
        name='buffalo_s',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    face_detector.prepare(ctx_id=0, det_size=DET_SIZE)
    print("  ✓ Face detector loaded.\n")

    # ── Initialize pipeline components ──
    smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)
    timeline = EmotionTimeline(smoother)
    ux_calc = UsabilityMetricsCalculator()

    # ── Open video ──
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open video '{VIDEO_PATH}'")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_sec = total_frames / video_fps if video_fps > 0 else 0
    frame_interval = max(1, int(video_fps / PROCESS_FPS))

    print(f"  Video FPS:      {video_fps:.1f}")
    print(f"  Total Frames:   {total_frames}")
    print(f"  Duration:       {total_duration_sec:.1f}s")
    print(f"  Processing every {frame_interval}th frame ({PROCESS_FPS} fps)")
    print()

    # ── Process video frame by frame ──
    print("  Processing video...")
    frame_idx = 0
    processed_count = 0
    faces_found = 0
    faces_missed = 0
    prev_emotions = None
    all_ux_events = []
    all_ux_metrics = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to match target PROCESS_FPS
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp_ms = int((frame_idx / video_fps) * 1000)

        # Step A: Detect face with SCRFD
        faces = face_detector.get(frame)

        if faces:
            faces_found += 1
            # Take the largest face (closest to camera)
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            bbox = face.bbox.astype(int)

            # Step B+C: Crop, resize, preprocess
            x1, y1, x2, y2 = bbox
            # Add small padding to crop (10%)
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                faces_missed += 1
                frame_idx += 1
                continue

            tensor = preprocess_face(face_crop).to(device)

            # Step D: Predict emotions
            with torch.no_grad():
                logits = emotion_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

            emotions = {EMOTIONS[i]: round(float(probs[i]), 4) for i in range(7)}
            confidence = float(probs.max())

            # Step E: Add to timeline (smoothing happens inside)
            timeline.add_frame(timestamp_ms, emotions, confidence)

            # Step F: Compute usability metrics for this frame
            ux_metrics = ux_calc.compute(emotions, confidence)
            all_ux_metrics.append(ux_metrics)

            # Step G: Detect events
            events = ux_calc.detect_events(emotions, confidence, prev_emotions)
            for event in events:
                event['timestamp_ms'] = timestamp_ms
                event['timestamp_str'] = _ms_to_timestr(timestamp_ms)
                all_ux_events.append(event)

            prev_emotions = emotions
        else:
            faces_missed += 1

        processed_count += 1

        # Progress bar
        if processed_count % 50 == 0:
            elapsed = time.time() - start_time
            progress = frame_idx / total_frames * 100
            print(f"    {progress:5.1f}% | {processed_count} frames | "
                  f"{_ms_to_timestr(timestamp_ms)} | {elapsed:.0f}s elapsed")

        frame_idx += 1

    cap.release()
    total_time = time.time() - start_time

    print(f"\n  ✓ Processing complete in {total_time:.1f}s")
    print(f"    Frames processed:  {processed_count}")
    print(f"    Faces detected:    {faces_found}")
    print(f"    Faces missed:      {faces_missed}")
    print(f"    Detection rate:    {faces_found / max(processed_count, 1):.1%}")

    if faces_found == 0:
        print("\n  ❌ No faces detected in video. Check video quality or DET_SIZE.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════
    # Step H: Build timeline segments
    # ══════════════════════════════════════════════════════════════
    print("\n  Building emotion segments...")
    segments = timeline.build_segments(min_segment_ms=MIN_SEGMENT_MS)

    print(f"  ✓ {len(segments)} segments identified:\n")
    for i, seg in enumerate(segments):
        start_str = _ms_to_timestr(seg['start_ms'])
        end_str = _ms_to_timestr(seg['end_ms'])
        dur = seg['duration_ms'] / 1000
        print(f"    Segment {i + 1:3d}:  {start_str} → {end_str}  "
              f"({dur:5.1f}s)  {seg['dominant_emotion']:12s}  "
              f"conf={seg['confidence']:.2f}")

    # ══════════════════════════════════════════════════════════════
    # Step I: Summary aggregation
    # ══════════════════════════════════════════════════════════════
    print("\n  Computing summary...")
    summary = aggregate_summary(timeline)

    print(f"\n  ── OVERALL SUMMARY ──")
    print(f"  Dominant Emotion:   {summary['dominant_emotion']} "
          f"({summary['dominant_confidence']:.1%})")
    print(f"  Avg Confidence:     {summary['avg_confidence']:.3f}")
    print(f"  Total Duration:     {summary['total_duration_ms'] / 1000:.1f}s")
    print(f"\n  Emotion Distribution:")
    for emotion in EMOTIONS:
        pct = summary['overall_emotions'][emotion]
        time_pct = summary['time_distribution'].get(emotion, 0)
        bar = "█" * int(pct * 40)
        print(f"    {emotion:12s}  prob={pct:.3f}  time={time_pct:.1%}  {bar}")

    # ══════════════════════════════════════════════════════════════
    # Usability metrics summary
    # ══════════════════════════════════════════════════════════════
    if all_ux_metrics:
        avg_ux = {}
        for key in all_ux_metrics[0]:
            values = [m[key] for m in all_ux_metrics]
            avg_ux[key] = round(float(np.mean(values)), 4)
            avg_ux[f'{key}_max'] = round(float(np.max(values)), 4)

        print(f"\n  ── USABILITY METRICS ──")
        print(f"  Frustration Index:  {avg_ux['frustration_index']:.3f}  "
              f"(peak: {avg_ux['frustration_index_max']:.3f})")
        print(f"  Confusion Score:    {avg_ux['confusion_score']:.3f}  "
              f"(peak: {avg_ux['confusion_score_max']:.3f})")
        print(f"  Engagement Level:   {avg_ux['engagement_level']:.3f}  "
              f"(peak: {avg_ux['engagement_level_max']:.3f})")
        print(f"  Satisfaction Score: {avg_ux['satisfaction_score']:.3f}  "
              f"(peak: {avg_ux['satisfaction_score_max']:.3f})")
        print(f"  Boredom Score:      {avg_ux['boredom_score']:.3f}  "
              f"(peak: {avg_ux['boredom_score_max']:.3f})")
        print(f"  Stress Level:       {avg_ux['stress_level']:.3f}  "
              f"(peak: {avg_ux['stress_level_max']:.3f})")
    else:
        avg_ux = {}

    # Events
    if all_ux_events:
        print(f"\n  ── USABILITY EVENTS ({len(all_ux_events)} detected) ──")
        for ev in all_ux_events[:20]:  # show first 20
            icon = {"frustration_spike": "😤", "confusion_detected": "😕",
                    "delight_moment": "😊", "emotion_shift": "🔄"}.get(ev['type'], "•")
            ts = ev.get('timestamp_str', '??:??')
            print(f"    {icon} [{ts}] {ev['type']} — {ev.get('detail', ev.get('from', '') + '→' + ev.get('to', ''))}")
        if len(all_ux_events) > 20:
            print(f"    ... and {len(all_ux_events) - 20} more events")
    else:
        print(f"\n  No usability events detected (thresholds not exceeded).")

    # ══════════════════════════════════════════════════════════════
    # Save outputs
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Saving results...")

    # 1. Full JSON results
    results = {
        'video': VIDEO_PATH,
        'config': {
            'process_fps': PROCESS_FPS,
            'smoothing_window': SMOOTHING_WINDOW,
            'min_segment_ms': MIN_SEGMENT_MS,
        },
        'processing': {
            'total_frames_processed': processed_count,
            'faces_detected': faces_found,
            'faces_missed': faces_missed,
            'detection_rate': round(faces_found / max(processed_count, 1), 4),
            'processing_time_sec': round(total_time, 2),
        },
        'summary': summary,
        'usability_metrics': avg_ux,
        'usability_events': all_ux_events,
        'segments': segments,
        # Include per-frame data (can be large — comment out if not needed)
        'frame_predictions': [
            {
                'timestamp_ms': f['timestamp_ms'],
                'smoothed_emotions': f['smoothed_emotions'],
                'dominant_emotion': f['dominant_emotion'],
                'confidence': f['confidence'],
            }
            for f in timeline.raw_frames
        ],
    }

    json_path = os.path.join(OUTPUT_DIR, "emotion_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Full results: {json_path}")

    # 2. Human-readable segments
    seg_path = os.path.join(OUTPUT_DIR, "emotion_segments.txt")
    with open(seg_path, 'w') as f:
        f.write(f"EMOTION TIMELINE — {VIDEO_PATH}\n")
        f.write(f"{'=' * 60}\n\n")
        for i, seg in enumerate(segments):
            s = _ms_to_timestr(seg['start_ms'])
            e = _ms_to_timestr(seg['end_ms'])
            f.write(f"  {i + 1:3d}.  {s} → {e}  ({seg['duration_ms'] / 1000:.1f}s)  "
                    f"{seg['dominant_emotion']}  conf={seg['confidence']:.2f}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Dominant overall: {summary['dominant_emotion']}\n")
    print(f"  ✓ Segments:     {seg_path}")

    # 3. Timeline chart
    chart_path = os.path.join(OUTPUT_DIR, "emotion_timeline_chart.png")
    save_timeline_chart(timeline, chart_path)

    print(f"\n{'=' * 65}")
    print(f"  ██ PIPELINE COMPLETE ██")
    print(f"{'=' * 65}")
    print(f"\n  Output files in '{OUTPUT_DIR}/':")
    print(f"    • emotion_results.json        — all data as JSON")
    print(f"    • emotion_segments.txt         — readable segment list")
    print(f"    • emotion_timeline_chart.png   — visual emotion chart")
    print()


# ── Helper ───────────────────────────────────────────────────────

def _ms_to_timestr(ms: int) -> str:
    """Convert milliseconds to MM:SS.s format."""
    total_sec = ms / 1000.0
    minutes = int(total_sec // 60)
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:04.1f}"


if __name__ == "__main__":
    main()
