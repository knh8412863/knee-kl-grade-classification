# Knee Osteoarthritis KL Grade Classification Project

## Domain Rules & Constraints
- Task: Classify knee X-ray images into KL Grades (0 to 4).
- Input: Grayscale or RGB knee joint X-ray images.
- Metric: Accuracy, Cohen's Kappa Score (Crucial for ordinal classification like KL Grade).
- Explainability: Grad-CAM generation is mandatory for every inference.

## Tech Stack & Code Style
- Framework: PyTorch, PyTorch Lightning (optional)
- Models: Timm library (ResNet, EfficientNet, ConvNeXt)
- Style: PEP8, Strict type hinting, Clear docstrings for medical auditing.
- Visualization: OpenCV, Matplotlib for Grad-CAM overlay.

## Commands
- Data Check: `python src/dataset_check.py`
- Train: `python src/train.py --config config/default.yaml`
- Evaluate: `python src/evaluate.py --model_path checkpoints/best.pth`
