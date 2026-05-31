import json
import os
import pickle
import random
import tarfile
import urllib.request
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageChops, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset


SEED = 42
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticCIFARLikeDataset(Dataset):
    """Deterministic CIFAR-shaped fallback with class-specific geometric patterns."""

    def __init__(self, size=5000, image_size=32, num_classes=10, train=True, transform=None):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.train = train
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = idx % self.num_classes
        rng = np.random.default_rng(SEED + idx + (0 if self.train else 100000))
        image = rng.normal(loc=80, scale=18, size=(self.image_size, self.image_size, 3))

        row = 3 + (label % 5) * 5
        col = 3 + (label // 5) * 10
        channel = label % 3

        if label < 5:
            image[row : row + 5, :, channel] += 120
        else:
            image[:, col : col + 5, channel] += 120

        if label % 2 == 0:
            image[8:24, 8:24, (channel + 1) % 3] += 55
        else:
            diag = np.arange(self.image_size)
            image[diag, diag, (channel + 1) % 3] += 120

        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class ArrayImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(int(self.labels[idx]), dtype=torch.long)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return ImageOps.mirror(image)
        return image


class RandomAffine:
    def __init__(self, degrees=15, translate=(0.12, 0.12), scale=(0.85, 1.15)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, image):
        width, height = image.size
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(self.scale[0], self.scale[1])
        max_dx = self.translate[0] * width
        max_dy = self.translate[1] * height
        dx = random.uniform(-max_dx, max_dx)
        dy = random.uniform(-max_dy, max_dy)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        scaled = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (width, height), tuple(int(v * 255) for v in CIFAR_MEAN))
        paste_x = int((width - new_width) / 2 + dx)
        paste_y = int((height - new_height) / 2 + dy)
        canvas.paste(scaled, (paste_x, paste_y))
        return canvas.rotate(angle, resample=Image.Resampling.BICUBIC)


class RandAugment:
    def __init__(self, num_ops=2, magnitude=9):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.ops = [
            self._rotate,
            self._translate_x,
            self._translate_y,
            self._brightness,
            self._contrast,
            self._color,
            self._sharpness,
            self._posterize,
            self._solarize,
        ]

    def __call__(self, image):
        for op in random.sample(self.ops, k=min(self.num_ops, len(self.ops))):
            image = op(image)
        return image

    @property
    def strength(self):
        return self.magnitude / 10.0

    def _signed(self, value):
        return value if random.random() < 0.5 else -value

    def _rotate(self, image):
        return image.rotate(self._signed(30 * self.strength), resample=Image.Resampling.BICUBIC)

    def _translate_x(self, image):
        offset = int(self._signed(image.size[0] * 0.33 * self.strength))
        return ImageChops.offset(image, offset, 0)

    def _translate_y(self, image):
        offset = int(self._signed(image.size[1] * 0.33 * self.strength))
        return ImageChops.offset(image, 0, offset)

    def _brightness(self, image):
        return ImageEnhance.Brightness(image).enhance(1.0 + self._signed(0.9 * self.strength))

    def _contrast(self, image):
        return ImageEnhance.Contrast(image).enhance(1.0 + self._signed(0.9 * self.strength))

    def _color(self, image):
        return ImageEnhance.Color(image).enhance(1.0 + self._signed(0.9 * self.strength))

    def _sharpness(self, image):
        return ImageEnhance.Sharpness(image).enhance(1.0 + self._signed(0.9 * self.strength))

    def _posterize(self, image):
        bits = max(4, 8 - int(4 * self.strength))
        return ImageOps.posterize(image, bits)

    def _solarize(self, image):
        threshold = int(256 * (1.0 - 0.8 * self.strength))
        return ImageOps.solarize(image, threshold)


class ToTensorNormalize:
    def __init__(self, mean=CIFAR_MEAN, std=CIFAR_STD):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def make_transforms():
    train_transform = Compose(
        [
            RandAugment(num_ops=2, magnitude=9),
            RandomAffine(degrees=15, translate=(0.12, 0.12), scale=(0.85, 1.15)),
            RandomHorizontalFlip(p=0.5),
            ToTensorNormalize(),
        ]
    )
    val_transform = Compose([ToTensorNormalize()])
    return train_transform, val_transform


def subset_dataset(dataset, max_items: int | None):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:max_items].tolist()
    return Subset(dataset, indices)


def _load_cifar_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch["labels"], dtype=np.int64)
    return data, labels


def load_cifar10_from_archive(data_dir):
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    archive_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(cifar_dir):
        if not os.path.exists(archive_path):
            print("Downloading CIFAR-10 archive...")
            urllib.request.urlretrieve(CIFAR_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_dir)

    train_images = []
    train_labels = []
    for batch_idx in range(1, 6):
        images, labels = _load_cifar_batch(os.path.join(cifar_dir, f"data_batch_{batch_idx}"))
        train_images.append(images)
        train_labels.append(labels)
    test_images, test_labels = _load_cifar_batch(os.path.join(cifar_dir, "test_batch"))
    return np.concatenate(train_images), np.concatenate(train_labels), test_images, test_labels


def build_datasets(data_dir, train_transform, val_transform):
    try:
        train_images, train_labels, val_images, val_labels = load_cifar10_from_archive(data_dir)
        train_dataset = ArrayImageDataset(train_images, train_labels, transform=train_transform)
        val_dataset = ArrayImageDataset(val_images, val_labels, transform=val_transform)
        dataset_name = "cifar10"
    except Exception as exc:
        print(f"CIFAR-10 unavailable, using synthetic fallback: {exc}")
        dataset_name = "synthetic_cifar_like_patterns"
        train_dataset = SyntheticCIFARLikeDataset(size=5000, train=True, transform=train_transform)
        val_dataset = SyntheticCIFARLikeDataset(size=1000, train=False, transform=val_transform)
    return dataset_name, train_dataset, val_dataset


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        return self.fc(torch.flatten(out, 1))


def build_resnet18(num_classes=10):
    return ResNet18CIFAR(num_classes=num_classes)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    ground_truth = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (batch_predictions == labels).sum().item()
        total += images.size(0)
        if not is_train:
            predictions.extend(batch_predictions.detach().cpu().tolist())
            ground_truth.extend(labels.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "predictions": predictions,
        "ground_truth": ground_truth,
    }


def main():
    seed_everything(SEED)

    working_dir = os.path.join(os.getcwd(), "working")
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = int(os.environ.get("IMGCLS_EPOCHS", "5"))
    batch_size = int(os.environ.get("IMGCLS_BATCH_SIZE", "128"))
    train_limit = int(os.environ.get("IMGCLS_TRAIN_LIMIT", "10000"))
    val_limit = int(os.environ.get("IMGCLS_VAL_LIMIT", "2000"))
    num_workers = int(os.environ.get("IMGCLS_NUM_WORKERS", "2"))

    train_transform, val_transform = make_transforms()
    dataset_name, train_dataset, val_dataset = build_datasets(data_dir, train_transform, val_transform)
    train_dataset = subset_dataset(train_dataset, train_limit)
    val_dataset = subset_dataset(val_dataset, val_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet18(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    experiment_data = {
        dataset_name: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    for epoch in range(epochs):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        experiment_data[dataset_name]["metrics"]["train"].append(train_stats["accuracy"])
        experiment_data[dataset_name]["metrics"]["val"].append(val_stats["accuracy"])
        experiment_data[dataset_name]["losses"]["train"].append(train_stats["loss"])
        experiment_data[dataset_name]["losses"]["val"].append(val_stats["loss"])
        experiment_data[dataset_name]["predictions"] = val_stats["predictions"]
        experiment_data[dataset_name]["ground_truth"] = val_stats["ground_truth"]

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_stats['loss']:.4f}, train_accuracy={train_stats['accuracy']:.4f}, "
            f"validation_loss={val_stats['loss']:.4f}, validation_accuracy={val_stats['accuracy']:.4f}"
        )

    metrics_summary = {
        "dataset": dataset_name,
        "model": "cifar_adapted_resnet18",
        "augmentations": ["RandAugment", "RandomAffine(rotate, translate, scale)", "RandomHorizontalFlip"],
        "epochs": epochs,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "final_validation_accuracy": experiment_data[dataset_name]["metrics"]["val"][-1],
        "final_validation_loss": experiment_data[dataset_name]["losses"]["val"][-1],
        "timestamp": datetime.now().isoformat(),
    }

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    with open(os.path.join(working_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics_summary": metrics_summary,
        },
        os.path.join(working_dir, "resnet18_randaugment_checkpoint.pt"),
    )

    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
