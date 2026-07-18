# 실험 기록 (Experiment Log)

모델링 진행 과정과 의사결정 근거를 시간 순으로 기록합니다.

---

## 2026-07-18 — Baseline 모델 설계 및 1차 학습 결과

### 1. 데이터
- 출처: [Mendeley KneeXrayData](https://data.mendeley.com/datasets/56rmx5bjcr/1) (`kneeKL224`, 224x224 grayscale PNG)
- 분포: train 5,778 / val 826 / test 1,656장
- 클래스 불균형 확인 (train 기준): grade 0=2,286 / 1=1,046 / 2=1,516 / 3=757 / 4=173
  - grade 4(가장 심함)가 grade 0(정상) 대비 약 13배 적음

### 2. 모델 설계 결정
- **백본**: `timm`의 `efficientnet_b0` (ImageNet 사전학습)
  - 입력은 1채널(그레이스케일) 또는 3채널(사전학습 가중치 재사용) 모두 지원하도록 옵션화
- **Loss**: `CrossEntropy + MSE(softmax 기대값, 정답 등급)` 결합 (`OrdinalCELoss`)
  - 이유: KL grade는 순서형(ordinal) 데이터라, grade 0을 4로 잘못 예측하는 것과 0을 1로 잘못 예측하는 것은 심각도가 다름. 일반 CE는 이를 구분하지 못해 MSE 항으로 "먼 오답"에 더 큰 페널티를 줌
- **클래스 불균형 대응**: `WeightedRandomSampler`로 train 샘플링 시 그레이드별 역빈도 가중치 적용
- **평가지표**: Accuracy + Cohen's Kappa(quadratic weighted) — 순서형 분류의 표준 지표

### 3. 개발/검증 과정
- 로컬(CPU 전용) 환경에서 배치당 약 25초 측정 → 10 epoch 풀 학습 시 약 12~13시간 예상
- 코드 정상 동작 검증을 위해 데이터 일부(train 300 / val 100, 3 epoch)로 스모크 테스트 먼저 진행 → loss 감소, accuracy 상승 확인 (파이프라인 정상)
- **버그 발견 및 수정**: 스모크 테스트를 GPU(Colab) 환경에서 실행했을 때 `RuntimeError: Expected all tensors to be on the same device` 발생
  - 원인: `src/train.py`에서 `model`은 `.to(device)` 했지만 `OrdinalCELoss` 내부 버퍼(`grades`)는 CPU에 남아있었음
  - 수정: `loss_fn = OrdinalCELoss(...).to(device)` 추가
- 본 학습은 Google Colab GPU 환경에서 `python src/train.py --config config/default.yaml` (10 epoch, 전체 데이터)로 진행

### 4. 1차 학습 결과 (val set, 826장 기준)
- **Accuracy: 0.557**
- **Cohen's Kappa (quadratic): 0.661** → Substantial agreement 구간 (Landis & Koch 기준 0.61~0.80)

Confusion Matrix (행=실제, 열=예측):

| 실제\예측 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 235 | 48 | 43 | 2 | 0 |
| 1 | 78 | 36 | 37 | 2 | 0 |
| 2 | 57 | 26 | 113 | 16 | 0 |
| 3 | 8 | 5 | 27 | 63 | 3 |
| 4 | 0 | 1 | 1 | 12 | 13 |

Grade별 precision/recall/f1:

| grade | precision | recall | f1 | support |
|---|---|---|---|---|
| 0 | 0.622 | 0.716 | 0.666 | 328 |
| 1 | 0.310 | 0.235 | 0.268 | 153 |
| 2 | 0.511 | 0.533 | 0.522 | 212 |
| 3 | 0.663 | 0.594 | 0.627 | 106 |
| 4 | 0.812 | 0.481 | 0.605 | 27 |

### 5. 분석
- 오분류가 대부분 **인접 등급 사이**(0↔1, 2↔3)에서 발생 — grade 0을 4로, 혹은 4를 0으로 예측한 사례는 0건. Ordinal loss 설계 의도대로 "먼 오답"은 회피되고 있음
- **grade 1의 recall이 가장 낮음(0.235)** — grade 0과의 경계 모호성이 주원인으로 추정 (임상적으로도 grade 0/1 구분은 판독자 간 일치도가 낮은 구간)
- **grade 4는 precision은 높지만 recall이 낮음(0.481)** — 학습 데이터 부족(173장)이 원인으로 추정. 다만 놓친 경우도 대부분 인접 등급(3)으로 예측되어 심각한 오류는 아님
- 참고 문헌상 이 데이터셋의 baseline 성능(accuracy 60~70%, kappa 0.6~0.7대)과 비교했을 때, 10 epoch만 학습한 1차 결과로는 준수한 수준

### 6. 다음 개선 방향 (계획)
- [ ] epoch 수 증가 (10 → 20~30) 및 learning rate scheduler 도입
- [ ] Data augmentation 추가 (random rotation, horizontal flip 등)
- [ ] grade 1 오분류 원인 분석을 위한 Grad-CAM 시각화 (관절 틈새/골극 부위를 실제로 보고 있는지 확인)
- [ ] grade 4 recall 개선을 위한 추가 augmentation 또는 focal loss 검토
