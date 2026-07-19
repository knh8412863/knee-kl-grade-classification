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
from fastapi.responses import FileResponse
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

# Fit on the validation set for the current baseline checkpoint (see
# "docs/modeling.md", 5th iteration). Re-run `src/evaluate.py` and update this if
# the checkpoint changes.
GRADE_THRESHOLDS = [0.85, 1.65, 2.5, 3.55]

GRADE_DESCRIPTIONS = {
    0: "정상 (No radiographic osteophytes or joint space narrowing)",
    1: "의심스러운 소견 (Doubtful joint space narrowing, possible osteophytic lipping)",
    2: "경도 (Definite osteophytes, possible joint space narrowing)",
    3: "중등도 (Multiple osteophytes, definite joint space narrowing, some sclerosis)",
    4: "중증 (Large osteophytes, marked joint space narrowing, severe sclerosis)",
}

# The model has no notion of "is this even an X-ray" — it was trained only on knee
# X-rays, so it will happily assign a confident-looking grade to a photo of a face
# or anything else. These two heuristics catch the obvious misuse cases (not a
# trained OOD detector, just cheap signals): true X-rays are near-grayscale even
# when stored as RGB, and a low max-probability suggests the model itself is
# unsure the input resembles anything it was trained on.
XRAY_COLOR_DIFF_THRESHOLD = 12.0
LOW_CONFIDENCE_THRESHOLD = 0.55

NOT_XRAY_MESSAGE = (
    "이 이미지는 무릎 X-ray가 아니거나 판독하기에 화질/각도가 적절하지 않은 것으로 보입니다. "
    "무릎 X-ray 원본 이미지로 다시 시도해 주세요. (자동 필터링 — 실제 X-ray인데 이 메시지가 "
    "뜬다면 조명을 줄이고 정면 각도로 다시 촬영해 보세요.)"
)


def looks_like_xray(image: Image.Image) -> bool:
    arr = np.asarray(image).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    channel_diff = (np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3
    return channel_diff < XRAY_COLOR_DIFF_THRESHOLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KLGradeModel(in_channels=IN_CHANNELS, pretrained=False, backbone=MODEL_BACKBONE)
model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
model.to(device).eval()

transform = build_transform(in_channels=IN_CHANNELS, augment=False)
genai_client = genai.Client()  # reads GOOGLE_API_KEY from the environment

app = FastAPI(
    title="Knee KL Grade Reading Assistant",
    description=(
        "무릎 X-ray 이미지를 업로드하면 KL grade(0~4) 예측, Grad-CAM 시각화, "
        "LLM 기반 한국어 판독 소견서를 반환합니다. 대화형 문서는 /docs, "
        "OpenAPI 스키마는 /openapi.json에서 확인할 수 있습니다."
    ),
    version="1.0.0",
)


def generate_report(grade: int, confidence: float) -> str:
    # Only the final grade + its own confidence are given to the LLM — passing the
    # full per-grade probability breakdown let it "notice" cases where a different
    # grade had a higher raw probability and second-guess the official grade in the
    # generated text, contradicting the badge shown in the UI.
    prompt = f"""당신은 정형외과 영상 판독을 보조하는 AI입니다. 아래 모델 예측 결과를 바탕으로
한국어 판독 소견문을 작성하세요. 등급, 신뢰도, 소견을 포함하고, 마지막에 "이 결과는 참고용
스크리닝 소견이며 확정 진단은 전문의 판독이 필요합니다"라는 안내 문구를 반드시 포함하세요.
주어진 등급을 그대로 신뢰하고, 다른 등급의 가능성을 언급하며 반박하지 마세요.

예측 등급: KL Grade {grade} ({GRADE_DESCRIPTIONS[grade]})
신뢰도: {confidence:.1%}

3~5문장으로 간결하게 작성하세요."""

    response = genai_client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return response.text


@app.get("/health", summary="서버 상태 확인", description="모델이 정상적으로 로드되어 응답 가능한 상태인지 확인합니다.")
def health() -> dict:
    return {"status": "ok", "device": str(device), "backbone": MODEL_BACKBONE}


@app.get("/camera", summary="카메라 실시간 판독 페이지", description="웹캠으로 X-ray를 비추면 자동으로 판독하는 HTML 페이지를 반환합니다.")
def camera_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "camera.html")


@app.post(
    "/predict",
    summary="X-ray 이미지 판독",
    description=(
        "무릎 X-ray 이미지 파일을 받아 KL grade(0~4), 신뢰도, Grad-CAM 히트맵(base64 PNG), "
        "LLM 판독 소견서를 반환합니다. 업로드된 이미지가 X-ray로 보이지 않거나 신뢰도가 "
        "낮으면(`likely_xray: false`) 등급 대신 확인 안내 메시지를 반환합니다."
    ),
)
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
    confidence = float(probs[predicted_grade])
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

    likely_xray = bool(looks_like_xray(pil_image)) and confidence >= LOW_CONFIDENCE_THRESHOLD
    report = generate_report(predicted_grade, confidence) if likely_xray else NOT_XRAY_MESSAGE

    return {
        "predicted_grade": predicted_grade,
        "grade_description": GRADE_DESCRIPTIONS[predicted_grade],
        "confidence": confidence,
        "grade_probabilities": prob_list,
        "gradcam_image_base64": gradcam_b64,
        "report": report,
        "likely_xray": likely_xray,
    }
