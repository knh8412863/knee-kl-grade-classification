"""Evaluates a trained checkpoint: plain argmax vs Kappa-optimized ordinal thresholds.

Thresholds are tuned on the validation set only, then applied unchanged to the test
set, so the reported test numbers reflect genuine generalization rather than fitting
to test data.
"""

import argparse

import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader

from baseline import KLGradeModel, KneeXrayDataset, NUM_CLASSES
from threshold_tuning import DEFAULT_THRESHOLDS, apply_thresholds, get_expected_grades, tune_thresholds


def report(name: str, targets, preds) -> None:
    accuracy = accuracy_score(targets, preds)
    kappa = cohen_kappa_score(targets, preds, weights="quadratic")
    print(f"--- {name} ---")
    print(f"accuracy={accuracy:.4f} kappa={kappa:.4f}")
    print(confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES))))
    print(classification_report(targets, preds, labels=list(range(NUM_CLASSES)), digits=3))
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = config["data"]["in_channels"]
    batch_size = config["train"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    model = KLGradeModel(in_channels=in_channels, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    val_loader = DataLoader(
        KneeXrayDataset(config["data"]["val_dir"], in_channels=in_channels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        KneeXrayDataset(config["data"]["test_dir"], in_channels=in_channels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_expected, val_targets = get_expected_grades(model, val_loader, device)
    test_expected, test_targets = get_expected_grades(model, test_loader, device)

    tuned_thresholds = tune_thresholds(val_expected, val_targets)
    print(f"tuned thresholds (fit on val): {[round(t, 3) for t in tuned_thresholds]}\n")

    report("test / standard rounding", test_targets, apply_thresholds(test_expected, DEFAULT_THRESHOLDS))
    report("test / Kappa-tuned thresholds", test_targets, apply_thresholds(test_expected, tuned_thresholds))


if __name__ == "__main__":
    main()
