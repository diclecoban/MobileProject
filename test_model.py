"""
Test the trained emotion detection model and provide detailed statistics
"""
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import time

print("PyTorch:", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------- DATA ----------
ROOT = "data/fer2013"
assert os.path.exists(f"{ROOT}/test"), f"Missing: {ROOT}/test"

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("\nLoading test dataset...")
test_ds = datasets.ImageFolder(f"{ROOT}/test", transform=tfm)
print(f"Classes ({len(test_ds.classes)}): {test_ds.classes}")
print(f"Test samples: {len(test_ds)}")

test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

# ---------- MODEL ----------
print("\nLoading trained model...")
num_classes = len(test_ds.classes)
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

model_path = Path(os.environ.get("LEGACY_MODEL_PATH", "artifacts/legacy/emotion_model.pth"))
# Load trained weights
if not model_path.exists():
    print(f"ERROR: {model_path} not found!")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded successfully from {model_path}!")

# ---------- EVALUATION ----------
print("\n" + "="*60)
print("TESTING MODEL ON TEST DATASET")
print("="*60)

all_preds = []
all_labels = []

start_time = time.time()
total_samples = 0

with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(test_loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_samples += labels.size(0)
        
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(test_loader)} ({total_samples}/{len(test_ds)} samples)")

test_time = time.time() - start_time

# ---------- STATISTICS ----------
print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)

# Overall Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
print(f"   Correctly predicted: {int(accuracy*len(all_labels))}/{len(all_labels)} samples")
print(f"   Test time: {test_time:.2f}s ({test_time/len(all_labels)*1000:.2f}ms per image)")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=range(num_classes)
)

print("\n" + "-"*60)
print("PER-CLASS STATISTICS")
print("-"*60)
print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
print("-"*60)

for i, class_name in enumerate(test_ds.classes):
    print(f"{class_name:<12} {precision[i]*100:>10.2f}%  {recall[i]*100:>10.2f}%  {f1[i]*100:>10.2f}%  {int(support[i]):>8}")

# Macro and Weighted averages
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='macro'
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='weighted'
)

print("-"*60)
print(f"{'Macro Avg':<12} {macro_precision*100:>10.2f}%  {macro_recall*100:>10.2f}%  {macro_f1*100:>10.2f}%")
print(f"{'Weighted Avg':<12} {weighted_precision*100:>10.2f}%  {weighted_recall*100:>10.2f}%  {weighted_f1*100:>10.2f}%")
print("-"*60)

# Confusion Matrix
print("\n" + "-"*60)
print("CONFUSION MATRIX")
print("-"*60)
cm = confusion_matrix(all_labels, all_preds)

# Print header
header = "True\\Pred".ljust(12)
for class_name in test_ds.classes:
    header += f"{class_name[:8]:<10}"
print(header)
print("-"*60)

# Print matrix rows
for i, class_name in enumerate(test_ds.classes):
    row = f"{class_name:<12}"
    for j in range(num_classes):
        row += f"{cm[i][j]:<10}"
    print(row)

# Detailed Classification Report
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=test_ds.classes, digits=4))

# Per-class accuracy
print("\n" + "-"*60)
print("PER-CLASS ACCURACY (Correct predictions per class)")
print("-"*60)
for i, class_name in enumerate(test_ds.classes):
    correct = cm[i][i]
    total = support[i]
    class_acc = (correct / total * 100) if total > 0 else 0
    print(f"{class_name:<12}: {correct}/{int(total)} ({class_acc:.2f}%)")

print("\n" + "="*60)
print("✅ TESTING COMPLETE")
print("="*60)
