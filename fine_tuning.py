import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from timm import create_model
from collections import Counter

# =========================================================
# DATASET
# =========================================================

class UsabilityEmotionDataset(Dataset):
    LABELS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 224):
        self.root = Path(root_dir) / split
        self.samples = []

        for label_idx, label_name in enumerate(self.LABELS):
            label_dir = self.root / label_name
            if not label_dir.exists():
                continue

            # ✅ support multiple formats
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in label_dir.glob(ext):
                    self.samples.append((str(img_path), label_idx))

        np.random.shuffle(self.samples)

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(label)


# =========================================================
# CLASS WEIGHTS
# =========================================================

def compute_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    total = sum(counts.values())
    num_classes = len(dataset.LABELS)

    weights = []
    for i in range(num_classes):
        weights.append(total / (num_classes * counts[i]))

    return torch.tensor(weights, dtype=torch.float)


# =========================================================
# MODEL
# =========================================================

class EmotiEffLibFineTuned(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.backbone = create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# =========================================================
# TRAINING
# =========================================================

def train_facial_emotion_model(dataset_dir: str, output_dir: str = './trained_model'):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    train_dataset = UsabilityEmotionDataset(dataset_dir, 'train')
    val_dataset = UsabilityEmotionDataset(dataset_dir, 'val')

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # ✅ Windows-safe
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=num_workers)

    model = EmotiEffLibFineTuned().to(device)

    # ✅ class weights
    class_weights = compute_class_weights(train_dataset).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    # =====================================================
    # PHASE 1
    # =====================================================
    model.freeze_backbone()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=0.01
    )

    print("\nPHASE 1")

    for epoch in range(5):
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion,
                                      None, device, False)

        print(f"Epoch {epoch+1} | Train {train_acc:.2f}% | Val {val_acc:.2f}%")

    # =====================================================
    # PHASE 2
    # =====================================================
    model.unfreeze_backbone()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15, eta_min=1e-7
    )

    best_val = 0
    patience = 0

    print("\nPHASE 2")

    for epoch in range(15):
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion,
                                      None, device, False)

        scheduler.step()

        print(f"Epoch {epoch+1} | Train {train_acc:.2f}% | Val {val_acc:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            patience = 0
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            print("Saved best model")
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping")
                break

    print("Best Val Acc:", best_val)
    return model


# =========================================================
# EPOCH LOOP
# =========================================================

def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total

if __name__ == "__main__":
    train_facial_emotion_model(
        dataset_dir="dataset",          # path to your dataset
        output_dir="./trained_model"    # where model will be saved
    )