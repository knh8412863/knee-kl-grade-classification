# FastAPI Reading-Report Service

무릎 X-ray 이미지를 업로드하면 KL grade 예측 + Grad-CAM + LLM 판독 소견서를 반환하는 웹 서비스.

## 구성

- `src/api.py`: FastAPI 앱. `/health`, `/predict` 엔드포인트
- 모델: 5차 baseline 체크포인트(`efficientnet_b3`, 299x299, threshold 튜닝) — 상세는 [`modeling.md`](modeling.md) 참고
- LLM: Google Gemini API (`gemini-flash-latest`)로 등급/신뢰도/소견이 담긴 한국어 판독 소견서 생성

## 실행 방법

```powershell
cd knee-kl-grade-classification
uvicorn api:app --app-dir src --port 8000
```

- `checkpoints/best.pth`에 실제 체크포인트 파일이 있어야 함 (git에는 포함 안 됨)
- `.env`에 `GOOGLE_API_KEY` 설정 필요

## `/predict` 응답 형식

```json
{
  "predicted_grade": 4,
  "grade_description": "중증 (Large osteophytes, ...)",
  "confidence": 0.98,
  "grade_probabilities": [0.0, 0.0, 0.0, 0.02, 0.98],
  "gradcam_image_base64": "...",
  "report": "정형외과 영상 AI 판독 보조 소견서..."
}
```

## 검증 완료

실제 서버 기동 + 실제 test set 이미지(grade 4)로 end-to-end 테스트 — 예측(grade 4, 신뢰도 98%) 정확, Grad-CAM 정상 생성, Gemini 리포트 정상 생성(전문의 확인 필요 문구 포함) 확인.

---

# Troubleshooting & Notes (API)

## 1. Google Gemini API 모델 버전/quota 문제

**문제**: LLM 리포트 생성 기능 구현 중, 처음 사용한 모델명이 순서대로 다 막힘.
1. `gemini-2.0-flash` → `429 RESOURCE_EXHAUSTED`, quota 자체가 0으로 설정된 상태 (신규 계정/무료 등급에서 이 모델은 완전히 비활성화)
2. `gemini-2.5-flash` → `404 NOT_FOUND`, "신규 사용자에게는 더 이상 제공되지 않는 모델"이라는 응답

**해결**: 특정 버전을 하드코딩하는 대신 **최신 flash 모델을 가리키는 별칭 `gemini-flash-latest`**를 사용하도록 변경. 실제로는 이 별칭이 `Gemini 3.5 Flash`로 연결됨 (2026-07-18 기준). Google AI Studio의 "비율 제한" 페이지에서 모델별 실제 사용 가능 여부(RPM 한도가 0인지 아닌지)를 먼저 확인하는 게 시행착오를 줄이는 방법.

**교훈**: LLM 제공사 모델은 세대교체가 빨라서, 프로덕션 코드에 특정 버전을 고정하기보다 `-latest` 류의 별칭을 쓰는 게 유지보수에 유리함.

---

## 2. 환경변수(API 키) 관리 방침

- `.env` 파일에 `GOOGLE_API_KEY=키값` 형태로 저장, `python-dotenv`로 로드
- `.env`는 처음부터 `.gitignore`에 포함되어 있어 커밋 누락 위험 없음

---

## 3. FastAPI 서버 테스트 시 Windows 환경 특이사항

- `uvicorn api:app --app-dir src`처럼 `--app-dir`를 지정하지 않고 `src/` 폴더 안에서 실행하면, 코드 내 상대경로(`checkpoints/best.pth`)가 실행 위치 기준으로 풀려서 `FileNotFoundError` 발생 — 항상 프로젝트 루트에서 `--app-dir src` 옵션으로 실행
- Git Bash에서 백그라운드로 띄운 uvicorn 프로세스를 재시작할 때, Git Bash의 `pkill`/`ps` 조합이 Windows 프로세스를 못 잡는 경우가 있어 **PowerShell의 `Get-Process`/`Stop-Process`로 종료하는 게 안정적**
- 콘솔에 한글 출력 시 Git Bash 기본 코드페이지 때문에 깨져 보이는 경우가 있음 (`chcp 65001` 또는 `PYTHONIOENCODING=utf-8`로 확인하면 실제 데이터는 정상인 경우가 많음) — 실제 파일/DB에 저장된 데이터가 깨진 건지, 터미널 출력만 깨진 건지 구분 필요
