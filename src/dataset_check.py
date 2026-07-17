"""Quick sanity check that the KL grade dataset loads and batches correctly."""

from pathlib import Path

from torch.utils.data import DataLoader

from baseline import KneeXrayDataset

DATA_ROOT = Path("data/raw/KneeXrayData/ClsKLData/kneeKL224")


def check_split(split: str) -> None:
    dataset = KneeXrayDataset(DATA_ROOT / split, in_channels=3)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(loader))
    print(f"[{split}] samples={len(dataset)} batch_shape={tuple(images.shape)} labels={labels.tolist()}")


def main() -> None:
    for split in ("train", "val", "test"):
        check_split(split)


if __name__ == "__main__":
    main()
