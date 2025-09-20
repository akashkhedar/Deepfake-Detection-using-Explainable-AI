import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

# =====================
# Config
# =====================
DATA_DIR = "dataset_mini"  # change to "dataset" for full
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "resnet18_best.pth"

# =====================
# Transforms (same as val)
# =====================
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# =====================
# Dataset + Loader
# =====================
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"), transform=test_transforms
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

class_names = test_dataset.classes
print("Classes:", class_names)

# =====================
# Load Model
# =====================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =====================
# Predictions
# =====================
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# =====================
# Metrics
# =====================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Switch model to evaluation mode
model.eval()

misclassified_images = []
misclassified_labels = []
misclassified_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:  # test_loader like val_loader
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Find wrong predictions
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified_images.append(inputs[i].cpu())
                misclassified_labels.append(labels[i].cpu())
                misclassified_preds.append(preds[i].cpu())


# Function to unnormalize image (to show properly)
def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


# Show first 12 mistakes
plt.figure(figsize=(12, 12))
for idx in range(12):
    plt.subplot(3, 4, idx + 1)
    true_label = class_names[misclassified_labels[idx]]
    pred_label = class_names[misclassified_preds[idx]]
    imshow(misclassified_images[idx], f"T: {true_label} | P: {pred_label}")
plt.tight_layout()
plt.show()

# Pick the layer to visualize (last conv layer of ResNet18)
cam_extractor = GradCAM(model, target_layer="layer4")


# Function to plot image with CAM overlay
def show_cam_on_image(img_tensor, cam, title=""):
    # Unnormalize image
    img = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Convert CAM to CPU and resize
    cam = cam.cpu()  # move to CPU
    if len(cam.shape) == 3:
        cam = cam.unsqueeze(0)  # add batch dim if needed

    # Upsample to image size
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

    # Remove batch and channel dims properly
    cam = cam.squeeze()  # removes all size-1 dims
    # cam should now be (224, 224)

    cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
    heatmap = plt.cm.jet(cam)[..., :3]  # apply colormap → shape (224,224,3)

    # Overlay heatmap on image
    overlay = 0.4 * heatmap + 0.6 * img
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")


# Show Grad-CAM for first 6 misclassified images
plt.figure(figsize=(12, 8))
for idx in range(6):
    input_img = misclassified_images[idx].unsqueeze(0).to(DEVICE)
    pred = misclassified_preds[idx].item()

    # Generate CAM
    out = model(input_img)
    cam = cam_extractor(pred, out)  # CAM for predicted class

    plt.subplot(2, 3, idx + 1)
    show_cam_on_image(
        misclassified_images[idx],
        cam[0],
        title=f"T: {class_names[misclassified_labels[idx]]} | P: {class_names[pred]}",
    )

plt.tight_layout()
plt.show()
