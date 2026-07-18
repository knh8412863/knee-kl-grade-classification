"""EfficientNet-B0 backbone, dataset loader, and ordinal-aware loss for KL grade classification."""

from pathlib import Path
from typing import Literal

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

NUM_CLASSES = 5


class KneeXrayDataset(Dataset):
    """Loads KL-grade labeled knee X-ray PNGs from a `<root_dir>/<grade>/*.png` layout."""

    def __init__(
        self,
        root_dir: str | Path,
        in_channels: Literal[1, 3] = 3,
        augment: bool = False,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.in_channels = in_channels
        self.samples: list[tuple[Path, int]] = []
        for grade_dir in sorted(self.root_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            label = int(grade_dir.name)
            self.samples.extend((p, label) for p in grade_dir.glob("*.png"))

        self.transform = transform or self._build_transform(augment)

    def _build_transform(self, augment: bool) -> transforms.Compose:
        mean = [0.5] * self.in_channels
        std = [0.5] * self.in_channels
        ops: list = []
        if augment:
            # Mild augmentations: knee X-rays are sensitive to intensity/orientation
            # shifts, so keep rotation and brightness/contrast jitter small enough
            # to avoid distorting the joint space or osteophyte features KL grading
            # depends on.
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        ops += [
            transforms.Grayscale(num_output_channels=self.in_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label


class KLGradeModel(nn.Module):
    """EfficientNet-B0 backbone with a KL-grade classification head."""

    def __init__(self, in_channels: Literal[1, 3] = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class OrdinalCELoss(nn.Module):
    """Cross-entropy combined with an MSE penalty on the softmax-expected grade.

    Plain cross-entropy scores predicting grade 1 and grade 4 as equally wrong
    when the true grade is 0, but grade 4 is a far worse clinical miss. The MSE
    term on the expected grade (softmax probabilities weighted by grade index)
    scales the penalty with how far the prediction is from the true grade.
    """

    def __init__(self, mse_weight: float = 0.5) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.register_buffer("grades", torch.arange(NUM_CLASSES, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1)
        expected_grade = probs @ self.grades
        mse = F.mse_loss(expected_grade, targets.float())
        return ce + self.mse_weight * mse
