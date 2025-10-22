import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# =====================
# Config
# =====================
DATA_DIR = "./dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SAVE_PATH = "resnet50_best.pth"
CHECKPOINT_PATH = (
    "#"  # separate from best model
)

# =====================
# Data Augmentations
# =====================
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def main():
    # =====================
    # Load Dataset
    # =====================
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "validation"), transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    class_names = train_dataset.classes
    print("Classes:", class_names)

    # =====================
    # Model Setup
    # =====================
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scaler = GradScaler("cuda")

    start_epoch = 0
    best_acc = 0.0

    # =====================
    # Resume from checkpoint if available
    # =====================
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Loading checkpoint from '{CHECKPOINT_PATH}' ...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"✅ Resumed from epoch {start_epoch} with best val acc: {best_acc:.4f}")

    # =====================
    # Training Loop
    # =====================
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)

        # Training
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # =====================
        # Save best model
        # =====================
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 Best model updated with acc: {best_acc:.4f}")

        # =====================
        # Save checkpoint after every epoch
        # =====================
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_acc": best_acc,
            },
            CHECKPOINT_PATH,
        )
        print(f"🧩 Checkpoint saved at epoch {epoch+1}")

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
