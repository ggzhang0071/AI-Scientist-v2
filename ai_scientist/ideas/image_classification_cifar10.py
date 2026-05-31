import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PatternImageDataset(Dataset):
    """Torch-only image dataset with class-specific geometric patterns."""

    def __init__(self, size=5000, image_size=32, num_classes=10, train=True):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = idx % self.num_classes
        generator = torch.Generator().manual_seed(SEED + idx + (0 if self.train else 100000))
        image = 0.12 * torch.randn(3, self.image_size, self.image_size, generator=generator)

        row = 3 + (label % 5) * 5
        col = 3 + (label // 5) * 10
        channel = label % 3

        if label < 5:
            image[channel, row : row + 5, :] += 1.0
        else:
            image[channel, :, col : col + 5] += 1.0

        if label % 2 == 0:
            image[(channel + 1) % 3, 8:24, 8:24] += 0.35
        else:
            diag = torch.arange(self.image_size)
            image[(channel + 1) % 3, diag, diag] += 0.8

        if self.train and random.random() < 0.5:
            image = torch.flip(image, dims=[2])

        image = image.clamp(-1.0, 1.5)
        return image, torch.tensor(label, dtype=torch.long)


def build_datasets():
    return (
        "synthetic_pattern_images",
        PatternImageDataset(size=5000, train=True),
        PatternImageDataset(size=1000, train=False),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


dataset_name, train_dataset, val_dataset = build_datasets()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

experiment_data = {
    dataset_name: {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

epochs = 5
for epoch in range(epochs):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion)

    experiment_data[dataset_name]["metrics"]["train"].append(train_acc)
    experiment_data[dataset_name]["metrics"]["val"].append(val_acc)
    experiment_data[dataset_name]["losses"]["train"].append(train_loss)
    experiment_data[dataset_name]["losses"]["val"].append(val_loss)

    print(
        f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
        f"train_accuracy={train_acc:.4f}, validation_loss={val_loss:.4f}, "
        f"validation_accuracy={val_acc:.4f}"
    )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
    },
    os.path.join(working_dir, "small_cnn_checkpoint.pt"),
)

print(f"Dataset: {dataset_name}")
print(f"Final Validation Accuracy: {experiment_data[dataset_name]['metrics']['val'][-1]:.4f}")
print(f"Final Validation Loss: {experiment_data[dataset_name]['losses']['val'][-1]:.4f}")
