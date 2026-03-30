"""
═══════════════════════════════════════════════════════════════════════
COMPLETE BENCHMARK SCRIPT — Facial Emotion Model
═══════════════════════════════════════════════════════════════════════

This script benchmarks your fine-tuned EmotiEffLib (EfficientNet-B0)
model on a test set and optionally compares against the old Keras
model2.h5.

HOW TO RUN:
    python benchmark_facial_emotion.py

BEFORE RUNNING — change these paths in the CONFIG section below:
    1. MODEL_PATH       → path to your fine-tuned best_model.pt
    2. TEST_DATASET_DIR → path to test/ folder (same structure as train/)
    3. OLD_MODEL_PATH   → (optional) path to old model2.h5 for comparison
    4. OUTPUT_DIR       → where to save benchmark results

REQUIRED PACKAGES:
    pip install torch torchvision timm scikit-learn matplotlib seaborn pillow numpy

TEST DATASET STRUCTURE:
    test/
    ├── Angry/
    │   ├── img001.jpg
    │   └── ...
    ├── Disgusted/
    ├── Fearful/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    └── Surprised/
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from timm import create_model

# Config

# Path to your fine-tuned model checkpoint (.pt file)
MODEL_PATH = "./fine-tuned-model/best_model.pt"

# Path to test dataset folder (must contain 7 emotion subfolders)
TEST_DATASET_DIR = "./dataset/val"

# (Optional) Path to old Keras model2.h5 for before/after comparison
# Set to None if you don't have it or don't want comparison
OLD_MODEL_PATH = "./fine-tuned-model/model2.h5"  # e.g. "./models/model2.h5"

# (Optional) Path to old test set formatted for old model (48x48 grayscale)
# Only needed if OLD_MODEL_PATH is set
OLD_TEST_DATASET_DIR = None  # e.g. "./dataset/test_48x48"

# Where to save benchmark outputs (confusion matrix image, JSON results)
OUTPUT_DIR = "./benchmark_results"

# Batch size for inference (increase if you have GPU memory)
BATCH_SIZE = 64

# Number of data loading workers (set to 0 on Windows if you get errors)
NUM_WORKERS = 0  # Windows-safe default; set to 4 on Linux/Mac

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']


# ── Model Definition (must match your training architecture) ─────

class EmotiEffLibFineTuned(nn.Module):
    """
    Same model class used during training.
    EfficientNet-B0 backbone + custom 7-class classifier head.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = create_model(
            'efficientnet_b0',
            pretrained=False,   # we load our own weights, not ImageNet
            num_classes=0,      # remove default head
        )
        feature_dim = self.backbone.num_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ── Test Dataset ─────────────────────────────────────────────────

class TestEmotionDataset(Dataset):
    """
    Loads test images from emotion-labeled subdirectories.
    No augmentation — clean evaluation only.
    """
    def __init__(self, root_dir: str, img_size: int = 224):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        root = Path(root_dir)
        for label_idx, label_name in enumerate(EMOTIONS):
            label_dir = root / label_name
            if not label_dir.exists():
                print(f"  ⚠ WARNING: {label_dir} not found, skipping '{label_name}'")
                continue
            count = 0
            for img_path in sorted(label_dir.glob('*')):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    self.samples.append((str(img_path), label_idx))
                    count += 1
            print(f"  ✓ {label_name}: {count} images")

        print(f"  Total test samples: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ── Core Benchmark Function ─────────────────────────────────────

def run_new_model_benchmark(model, test_loader, device):
    """
    Run full benchmark on the fine-tuned EmotiEffLib model.
    Returns dict of all metrics.
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, f1_score
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)

            # Measure latency per batch
            start_time = time.time()
            logits = model(images)
            elapsed = time.time() - start_time

            # Per-image latency
            latencies.append(elapsed / images.size(0))

            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            # Progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Batch {batch_idx + 1}/{len(test_loader)}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── 1. Classification Report ──
    print("\n" + "=" * 65)
    print("  CLASSIFICATION REPORT — Fine-Tuned EmotiEffLib (EfficientNet-B0)")
    print("=" * 65)
    report_str = classification_report(
        all_labels, all_preds,
        target_names=EMOTIONS,
        digits=4,
        zero_division=0
    )
    print(report_str)

    # ── 2. Key Metrics ──
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    avg_latency_ms = np.mean(latencies) * 1000
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0

    print("─" * 50)
    print(f"  Overall Accuracy:       {accuracy:.1%}")
    print(f"  Weighted F1 Score:      {f1_weighted:.4f}")
    print(f"  Macro F1 Score:         {f1_macro:.4f}")
    print(f"  Avg Latency:            {avg_latency_ms:.1f} ms/frame")
    print(f"  Throughput:             {fps:.0f} FPS")
    print(f"  Device:                 {device}")
    print("─" * 50)

    print("\n  Per-class F1 scores:")
    for i, emotion in enumerate(EMOTIONS):
        bar = "█" * int(per_class_f1[i] * 40)
        print(f"    {emotion:12s}  {per_class_f1[i]:.4f}  {bar}")

    # ── 3. Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    _save_confusion_matrix(cm, "Fine-Tuned EmotiEffLib", "confusion_matrix_new.png")

    # ── 4. Per-class accuracy ──
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # ── 5. Confidence analysis ──
    max_probs = np.max(all_probs, axis=1)
    correct_mask = all_preds == all_labels
    avg_conf_correct = np.mean(max_probs[correct_mask]) if correct_mask.sum() > 0 else 0
    avg_conf_wrong = np.mean(max_probs[~correct_mask]) if (~correct_mask).sum() > 0 else 0

    print(f"\n  Confidence Analysis:")
    print(f"    Avg confidence on CORRECT predictions: {avg_conf_correct:.3f}")
    print(f"    Avg confidence on WRONG predictions:   {avg_conf_wrong:.3f}")
    print(f"    → Gap: {avg_conf_correct - avg_conf_wrong:.3f} "
          f"(larger gap = better calibrated model)")

    # ── Compile results dict ──
    results = {
        'model': 'EmotiEffLib_FineTuned_EfficientNetB0',
        'test_samples': len(all_labels),
        'accuracy': round(float(accuracy), 4),
        'f1_weighted': round(float(f1_weighted), 4),
        'f1_macro': round(float(f1_macro), 4),
        'per_class_f1': {EMOTIONS[i]: round(float(per_class_f1[i]), 4) for i in range(7)},
        'per_class_accuracy': {EMOTIONS[i]: round(float(per_class_acc[i]), 4) for i in range(7)},
        'latency_ms': round(float(avg_latency_ms), 2),
        'fps': round(float(fps), 1),
        'device': str(device),
        'confidence_correct': round(float(avg_conf_correct), 4),
        'confidence_wrong': round(float(avg_conf_wrong), 4),
        'classification_report': report_str,
    }

    return results


# ── Old Model (model2.h5) Benchmark ─────────────────────────────

def run_old_model_benchmark(old_model_path, old_test_dir):
    """
    Benchmark the OLD Keras model2.h5 for before/after comparison.
    
    The old model expects: 48×48 grayscale images.
    If you don't have test images in that format, this function
    will preprocess from the same test dir (convert to grayscale, 
    resize to 48×48).
    """
    try:
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, f1_score, classification_report
    except ImportError:
        print("  ⚠ TensorFlow not installed. Skipping old model benchmark.")
        print("    Install with: pip install tensorflow")
        return None

    print("\nLoading old Keras model...")
    old_model = tf.keras.models.load_model(old_model_path, compile=False)

    # If no separate 48x48 dir, convert from main test dir
    test_dir = Path(old_test_dir) if old_test_dir else Path(TEST_DATASET_DIR)

    old_preds = []
    old_labels = []

    import cv2

    for label_idx, label_name in enumerate(EMOTIONS):
        label_dir = test_dir / label_name
        if not label_dir.exists():
            continue
        for img_path in sorted(label_dir.glob('*')):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                continue

            # Old model preprocessing: 48×48 grayscale, normalized to [0, 1]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=(0, -1))  # (1, 48, 48, 1)

            pred = old_model.predict(img, verbose=0)
            old_preds.append(np.argmax(pred[0]))
            old_labels.append(label_idx)

    if not old_preds:
        print("  ⚠ No images processed for old model.")
        return None

    old_accuracy = accuracy_score(old_labels, old_preds)
    old_f1 = f1_score(old_labels, old_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 65)
    print("  CLASSIFICATION REPORT — Old Model (model2.h5 / Keras CNN)")
    print("=" * 65)
    print(classification_report(
        old_labels, old_preds,
        target_names=EMOTIONS,
        digits=4,
        zero_division=0
    ))

    return {
        'model': 'old_model2_keras_cnn',
        'accuracy': round(float(old_accuracy), 4),
        'f1_weighted': round(float(old_f1), 4),
    }


# ── Confusion Matrix Plotter ────────────────────────────────────

def _save_confusion_matrix(cm, title, filename):
    """Save a confusion matrix heatmap as an image."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend (no GUI needed)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  ⚠ matplotlib/seaborn not installed. Skipping confusion matrix plot.")
        print("    Install with: pip install matplotlib seaborn")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTIONS, yticklabels=EMOTIONS,
        ax=ax, linewidths=0.5
    )
    ax.set_title(f'Confusion Matrix — {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Emotion', fontsize=12)
    ax.set_ylabel('True Emotion', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ Confusion matrix saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# ██  MAIN — ENTRY POINT  ██
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  FACIAL EMOTION MODEL — COMPLETE BENCHMARK")
    print("=" * 65)
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Test Set: {TEST_DATASET_DIR}")
    print(f"  Output:   {OUTPUT_DIR}")
    print()

    # ── Validate paths ──
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at '{MODEL_PATH}'")
        print(f"   → Update MODEL_PATH in the CONFIG section at the top of this script.")
        sys.exit(1)

    if not os.path.isdir(TEST_DATASET_DIR):
        print(f"❌ ERROR: Test dataset directory not found at '{TEST_DATASET_DIR}'")
        print(f"   → Update TEST_DATASET_DIR in the CONFIG section at the top.")
        print(f"   → Directory must contain subfolders: {', '.join(EMOTIONS)}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Setup device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    # ── Load model ──
    print("Loading fine-tuned EmotiEffLib model...")
    model = EmotiEffLibFineTuned(num_classes=7)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("  ✓ Model loaded successfully.\n")

    # ── Load test data ──
    print("Loading test dataset...")
    test_dataset = TestEmotionDataset(TEST_DATASET_DIR)

    if len(test_dataset) == 0:
        print("❌ ERROR: No test images found. Check your directory structure.")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )

    # ══════════════════════════════════════════════════════════════
    # RUN NEW MODEL BENCHMARK
    # ══════════════════════════════════════════════════════════════
    new_results = run_new_model_benchmark(model, test_loader, device)

    # ══════════════════════════════════════════════════════════════
    # RUN OLD MODEL BENCHMARK (optional)
    # ══════════════════════════════════════════════════════════════
    old_results = None
    if OLD_MODEL_PATH and os.path.exists(OLD_MODEL_PATH):
        print("\n" + "=" * 65)
        print("  BEFORE/AFTER COMPARISON — Old → New Model")
        print("=" * 65)
        old_results = run_old_model_benchmark(OLD_MODEL_PATH, OLD_TEST_DATASET_DIR)
    elif OLD_MODEL_PATH:
        print(f"\n  ⚠ Old model not found at '{OLD_MODEL_PATH}', skipping comparison.")

    # ══════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════════
    if old_results:
        print("\n" + "=" * 65)
        print("  ██ FINAL COMPARISON — BEFORE vs AFTER ██")
        print("=" * 65)
        print(f"  {'Metric':<25} {'Old (model2.h5)':<20} {'New (EmotiEffLib)':<20} {'Δ Change'}")
        print(f"  {'─'*25} {'─'*20} {'─'*20} {'─'*15}")

        old_acc = old_results['accuracy']
        new_acc = new_results['accuracy']
        delta_acc = new_acc - old_acc

        old_f1 = old_results['f1_weighted']
        new_f1 = new_results['f1_weighted']
        delta_f1 = new_f1 - old_f1

        print(f"  {'Accuracy':<25} {old_acc:<20.1%} {new_acc:<20.1%} {'+' if delta_acc >= 0 else ''}{delta_acc:.1%}")
        print(f"  {'Weighted F1':<25} {old_f1:<20.4f} {new_f1:<20.4f} {'+' if delta_f1 >= 0 else ''}{delta_f1:.4f}")
        print(f"  {'Throughput':<25} {'N/A':<20} {new_results['fps']:<20.0f} FPS")
        print(f"  {'Latency':<25} {'N/A':<20} {new_results['latency_ms']:<20.1f} ms")
        print()

        if delta_acc > 0:
            print(f"  ✅ New model is {delta_acc:.1%} more accurate than the old model!")
        else:
            print(f"  ⚠ New model did not improve over old model. Review training.")

        new_results['comparison'] = {
            'old_accuracy': old_acc,
            'new_accuracy': new_acc,
            'accuracy_improvement': round(float(delta_acc), 4),
            'old_f1': old_f1,
            'new_f1': new_f1,
            'f1_improvement': round(float(delta_f1), 4),
        }

    # ══════════════════════════════════════════════════════════════
    # SAVE RESULTS TO JSON
    # ══════════════════════════════════════════════════════════════
    # Remove non-serializable fields
    results_to_save = {k: v for k, v in new_results.items() if k != 'classification_report'}
    results_to_save['classification_report_text'] = new_results.get('classification_report', '')

    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n  ✓ Full results saved to: {results_path}")

    # Also save classification report as text
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("FACIAL EMOTION MODEL BENCHMARK REPORT\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test Set: {TEST_DATASET_DIR}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 65 + "\n\n")
        f.write(new_results.get('classification_report', ''))
        f.write(f"\nAccuracy: {new_results['accuracy']:.4f}\n")
        f.write(f"Weighted F1: {new_results['f1_weighted']:.4f}\n")
        f.write(f"Macro F1: {new_results['f1_macro']:.4f}\n")
        f.write(f"Latency: {new_results['latency_ms']:.1f} ms/frame\n")
        f.write(f"Throughput: {new_results['fps']:.0f} FPS\n")
    print(f"  ✓ Classification report saved to: {report_path}")

    print("\n" + "=" * 65)
    print("  ██ BENCHMARK COMPLETE ██")
    print("=" * 65)
    print(f"\n  Output files in '{OUTPUT_DIR}/':")
    print(f"    • benchmark_results.json      — all metrics as JSON")
    print(f"    • classification_report.txt   — human-readable report")
    print(f"    • confusion_matrix_new.png    — confusion matrix heatmap")
    print()


if __name__ == "__main__":
    main()
