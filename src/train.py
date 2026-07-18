"""Training loop for KL grade classification with imbalance-aware sampling and Cohen's Kappa evaluation."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from baseline import KLGradeModel, KneeXrayDataset, OrdinalCELoss, NUM_CLASSES


def subsample(dataset: KneeXrayDataset, max_samples: int | None) -> None:
    """Randomly truncates dataset.samples in place to at most `max_samples` entries."""
    if max_samples is None or len(dataset.samples) <= max_samples:
        return
    dataset.samples = random.sample(dataset.samples, max_samples)


def build_sampler(dataset: KneeXrayDataset) -> WeightedRandomSampler:
    """Weights each sample by the inverse frequency of its grade to counter class imbalance."""
    labels = np.array([label for _, label in dataset.samples])
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate(model: KLGradeModel, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds, weights="quadratic")
    return accuracy, kappa


def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = config["data"]["in_channels"]

    train_dataset = KneeXrayDataset(config["data"]["train_dir"], in_channels=in_channels)
    val_dataset = KneeXrayDataset(config["data"]["val_dir"], in_channels=in_channels)
    subsample(train_dataset, config["data"].get("max_train_samples"))
    subsample(val_dataset, config["data"].get("max_val_samples"))

    sampler = build_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        sampler=sampler,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    model = KLGradeModel(in_channels=in_channels, pretrained=True).to(device)
    loss_fn = OrdinalCELoss(mse_weight=config["train"]["mse_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    checkpoint_dir = Path(config["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config["checkpoint"]["filename"]

    best_kappa = -1.0
    for epoch in range(config["train"]["epochs"]):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)
        val_accuracy, val_kappa = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1}/{config['train']['epochs']} "
            f"train_loss={train_loss:.4f} val_accuracy={val_accuracy:.4f} val_kappa={val_kappa:.4f}"
        )

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved new best checkpoint to {checkpoint_path} (kappa={best_kappa:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
