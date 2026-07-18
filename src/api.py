"""FastAPI service: upload a knee X-ray, get a KL grade prediction, Grad-CAM overlay,
and an LLM-generated natural-language reading report.

Run with: uvicorn api:app --reload --app-dir src
"""

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from PIL import Image

from baseline import KLGradeModel, NUM_CLASSES, build_transform
from explainability import generate_gradcam
from threshold_tuning import apply_thresholds

load_dotenv()

MODEL_CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", "checkpoints/best.pth")
MODEL_BACKBONE = os.environ.get("MODEL_BACKBONE", "efficientnet_b3")
IN_CHANNELS = 3
INPUT_SIZE = 299

# Fit on the validation set for the current baseline checkpoint (see EXPERIMENTS.md,
# 5th iteration). Re-run `src/evaluate.py` and update this if the checkpoint changes.
GRADE_THRESHOLDS = [0.85, 1.65, 2.5, 3.55]

GRADE_DESCRIPTIONS = {
    0: "정상 (No radiographic osteophytes or joint space narrowing)",
    1: "의심스러운 소견 (Doubtful joint space narrowing, possible osteophytic lipping)",
    2: "경도 (Definite osteophytes, possible joint space narrowing)",
    3: "중등도 (Multiple osteophytes, definite joint space narrowing, some sclerosis)",
    4: "중증 (Large osteophytes, marked joint space narrowing, severe sclerosis)",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KLGradeModel(in_channels=IN_CHANNELS, pretrained=False, backbone=MODEL_BACKBONE)
model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
model.to(device).eval()

transform = build_transform(in_channels=IN_CHANNELS, augment=False)
genai_client = genai.Client()  # reads GOOGLE_API_KEY from the environment

app = FastAPI(title="Knee KL Grade Reading Assistant")


def generate_report(grade: int, confidence: float, probs: list[float]) -> str:
    prob_lines = "\n".join(f"- Grade {i}: {p:.1%}" for i, p in enumerate(probs))
    prompt = f"""당신은 정형외과 영상 판독을 보조하는 AI입니다. 아래 모델 예측 결과를 바탕으로
한국어 판독 소견문을 작성하세요. 등급, 신뢰도, 소견을 포함하고, 마지막에 "이 결과는 참고용
스크리닝 소견이며 확정 진단은 전문의 판독이 필요합니다"라는 안내 문구를 반드시 포함하세요.

예측 등급: KL Grade {grade} ({GRADE_DESCRIPTIONS[grade]})
등급별 확률 분포:
{prob_lines}

3~5문장으로 간결하게 작성하세요."""

    response = genai_client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return response.text


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(device), "backbone": MODEL_BACKBONE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    original = np.asarray(pil_image).astype(np.float32) / 255.0
    image_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = F.softmax(logits, dim=1)[0]
        grades = torch.arange(NUM_CLASSES, dtype=torch.float32)
        expected_grade = float((probs.cpu() * grades).sum())

    predicted_grade = int(apply_thresholds(np.array([expected_grade]), GRADE_THRESHOLDS)[0])
    confidence = float(probs.max())
    prob_list = probs.tolist()

    gradcam_path = generate_gradcam(
        model,
        image_tensor,
        original,
        true_grade=predicted_grade,
        save_name=f"{Path(file.filename).stem}_gradcam.png",
        device=device,
    )
    gradcam_b64 = base64.b64encode(gradcam_path.read_bytes()).decode("utf-8")

    report = generate_report(predicted_grade, confidence, prob_list)

    return {
        "predicted_grade": predicted_grade,
        "grade_description": GRADE_DESCRIPTIONS[predicted_grade],
        "confidence": confidence,
        "grade_probabilities": prob_list,
        "gradcam_image_base64": gradcam_b64,
        "report": report,
    }
