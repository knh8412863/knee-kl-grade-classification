# Knee X-ray KL Grade Classification

무릎 X-ray 영상에서 골관절염 중증도(KL grade 0~4)를 자동으로 분류하는 모델.

## 데이터셋

- 출처: [Mendeley Data](https://data.mendeley.com/datasets/56rmx5bjcr/1)
- `data/raw/` 에 압축을 풀어 저장 (git에는 포함하지 않음)

## 프로젝트 구조

```
data/
  raw/            # 원본 데이터셋
  processed/      # 전처리된 데이터
src/              # 학습/추론/전처리 코드
notebooks/        # 실험용 노트북
models/           # 학습된 모델 가중치 (git 미포함)
outputs/          # 결과물 (Grad-CAM 이미지 등, git 미포함)
```

## 진행 단계

1. [ ] 데이터 전처리 및 EDA
2. [ ] CNN(ResNet/EfficientNet) 파인튜닝
3. [ ] Grad-CAM 시각화
4. [ ] (확장) FastAPI 서비스화
