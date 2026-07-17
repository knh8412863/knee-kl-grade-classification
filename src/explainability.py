"""Grad-CAM generation for KL grade predictions, overlaid on the source X-ray image."""

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from baseline import KLGradeModel

OUTPUT_DIR = Path("outputs/gradcam")


def generate_gradcam(
    model: KLGradeModel,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    true_grade: int,
    save_name: str,
    device: torch.device | None = None,
) -> Path:
    """Overlays a Grad-CAM heatmap for the predicted KL grade onto the original image.

    Args:
        model: trained KLGradeModel in eval mode.
        image_tensor: preprocessed input, shape (1, C, H, W), already on `device`.
        original_image: HxWx3 float image in [0, 1] used as the overlay background.
        true_grade: ground-truth KL grade, shown in the saved figure's title.
        save_name: output filename (without directory), saved under `outputs/gradcam/`.
        device: device the model and tensor live on; defaults to the tensor's device.

    Returns:
        Path to the saved overlay image.
    """
    device = device or image_tensor.device
    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    target_layers = [model.backbone.conv_head]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        logits = model(image_tensor)
        pred_grade = int(logits.argmax(dim=1).item())

        grayscale_cam = cam(input_tensor=image_tensor)[0]

    overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    title = f"pred={pred_grade} true={true_grade}"
    overlay = cv2.copyMakeBorder(overlay, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(overlay, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / save_name
    cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return save_path
