# 배포 (Render, 무료)

카메라 실시간 판독 기능을 스마트폰에서 테스트하려면 HTTPS가 필요한데(로컬 `localhost`는 개발용으로만 가능), 가장 간단한 방법은 FastAPI 서비스를 무료로 배포하는 것. Java 관리자 서버 + MySQL은 별도 기능이라 이번 배포 범위에서 제외 — Python FastAPI(모델 + Grad-CAM + 카메라 페이지 + LLM 리포트)만 배포.

## 준비한 것

- `requirements-api.txt`: 서빙에 필요한 최소 의존성만 모음 (학습/평가 전용 패키지인 pandas, scikit-learn, matplotlib 등 제외 — 빌드 시간/용량 절약). PyTorch는 CPU 전용 wheel(`--extra-index-url .../whl/cpu`)을 사용해서 CUDA 버전(수 GB) 다운로드를 피함
- `model/best.pth`: 학습된 체크포인트를 git에 직접 커밋 (43MB, GitHub 100MB 제한 이내). 지금까지 `checkpoints/`는 git에서 제외해왔지만, 배포용 아티팩트는 재현성을 위해 예외로 커밋
- `render.yaml`: Render Blueprint 설정 — 빌드/실행 커맨드와 환경변수를 코드로 정의

## 배포 방법 (직접 해야 하는 부분)

1. [render.com](https://render.com) 가입 (GitHub 계정으로 로그인 가능)
2. 대시보드에서 "New" → "Blueprint" → 이 GitHub 저장소 연결 → `render.yaml` 자동 인식
3. 배포 중 `GOOGLE_API_KEY` 환경변수 입력 요청이 뜨면, `.env`에 넣어둔 값과 동일한 Gemini API 키 입력 (Render 대시보드에만 저장되고 git에는 들어가지 않음)
4. 배포 완료되면 `https://knee-kl-grade-api.onrender.com` 같은 주소가 생성됨 — 이 주소 + `/camera`로 스마트폰에서 접속

## 코드에서 함께 고친 것

- `threshold_tuning.py`가 최상단에서 `scikit-learn`을 import하고 있었는데, 실제로는 `tune_thresholds()`(학습/평가 시에만 사용) 함수 안에서만 필요함. `api.py`는 이 함수를 안 쓰는데도 모듈 전체를 import하는 순간 scikit-learn이 강제로 딸려 들어오고 있었음 → `import`를 함수 내부로 옮겨서 배포 시 불필요한 무거운 의존성(scikit-learn과 그 하위 의존성인 scipy 등) 제거

## 알려진 제약사항 (무료 티어)

- **콜드 스타트**: Render 무료 플랜은 일정 시간 요청이 없으면 서버가 슬립 상태로 전환됨. 슬립 후 첫 요청은 서버가 다시 켜지는 데 30초~1분 정도 걸릴 수 있음
- **메모리(RAM) 제한**: 무료 플랜은 512MB RAM. PyTorch + EfficientNet-B3 모델을 올리기엔 넉넉하지 않아서, 트래픽이 몰리면 메모리 부족으로 재시작될 수 있음 — 데모/포트폴리오 용도로는 문제없지만 실서비스 수준의 안정성은 아님
- Java 관리자 서버 + MySQL은 이번 배포에 포함되지 않음 (필요 시 별도로 진행)
