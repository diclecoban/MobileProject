# --- VERBOSE TRAIN (MobilenetV3 + FER2013) ---
import os, time, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

print("PyTorch:", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, flush=True)

LEGACY_ARTIFACT_DIR = Path(os.environ.get("LEGACY_ARTIFACT_DIR", "artifacts/legacy"))
LEGACY_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_MODEL_PATH = LEGACY_ARTIFACT_DIR / "emotion_model.pth"

# ---------- DATA ----------
ROOT = "data/fer2013"
assert os.path.exists(f"{ROOT}/train"), f"Missing: {ROOT}/train"
assert os.path.exists(f"{ROOT}/test"),  f"Missing: {ROOT}/test"

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

print("Loading ImageFolder...", flush=True)
train_ds = datasets.ImageFolder(f"{ROOT}/train", transform=tfm)
val_ds   = datasets.ImageFolder(f"{ROOT}/test",  transform=tfm)
print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}", flush=True)
print(f"Samples -> train: {len(train_ds)}, val: {len(val_ds)}", flush=True)
assert len(train_ds) > 0 and len(val_ds) > 0, "Empty dataset folders."

# hızlı ilerlemek için istersen küçük bir altkümeyle başla:
USE_SMALL = False  # True yaparsan her sınıftan az örnekle dener
if USE_SMALL:
    from collections import defaultdict
    keep_per_class = 300  # her sınıftan 300 örnek
    idxs = defaultdict(list)
    for i, (_, y) in enumerate(train_ds.samples):
        if len(idxs[y]) < keep_per_class:
            idxs[y].append(i)
    small_idx = sum(idxs.values(), [])
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, small_idx)
    print("Using small subset:", len(train_ds), "samples", flush=True)

print("Creating DataLoaders...", flush=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
print("DataLoaders ready.", flush=True)

# ---------- MODEL ----------
print("Building model (loading pretrained from cache, no progress bar)...", flush=True)
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = models.mobilenet_v3_small(weights=weights, progress=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(train_ds.classes))
model = model.to(device)
print("Model ready.", flush=True)

crit = nn.CrossEntropyLoss()
opt  = optim.Adam(model.parameters(), lr=1e-4)

# ---------- TRAIN ----------
epochs = 1  # önce 1 epoch ile akışı doğrula
for ep in range(epochs):
    t0 = time.time()
    model.train()
    run_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        out = model(imgs)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if i % 50 == 0:
            print(f"[train] epoch {ep+1} step {i}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
    print(f"Epoch {ep+1} done. avg_loss={run_loss/len(train_loader):.4f} time={time.time()-t0:.1f}s", flush=True)

# ---------- SAVE ----------
torch.save(model.state_dict(), LEGACY_MODEL_PATH)
print(f"✅ Saved: {LEGACY_MODEL_PATH}", flush=True)
