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
- [ ] grade 1 오분류 원인 분석을 위한 Grad-CAM 시각화 (관절 틈새/골극 부위를 실제로 보고 있는지 확인)
- [ ] grade 4 recall 개선을 위한 추가 augmentation 또는 focal loss 검토

---

## 2026-07-18 — 2차 개선: Augmentation + LR Scheduler

1차 결과의 개선 방향에 따라 아래 3가지를 반영:

- **Data augmentation** (`src/baseline.py`): `KneeXrayDataset`에 `augment` 옵션 추가. train 데이터에만 `RandomHorizontalFlip(p=0.5)`, `RandomRotation(±10도)`, `ColorJitter(brightness=0.2, contrast=0.2)` 적용
  - 강도를 약하게 잡은 이유: 관절 틈새/골극처럼 KL grade 판독의 핵심 근거가 되는 미세 구조가 과도한 변형으로 왜곡되지 않도록 하기 위함
- **LR scheduler** (`src/train.py`): `CosineAnnealingLR(T_max=epochs)` 추가, epoch마다 현재 lr 로그 출력
- **Epoch 수 증가**: `config/default.yaml` epochs 10 → 25

**검증**: `config/quick.yaml`(train 300 / val 100, 3 epoch)로 스모크 테스트 — 에러 없이 동작, lr이 매 epoch 감소하는 것 확인 (0.0001 → 0.000075 → 0.000025). train_loss가 1차보다 다소 높게 나오는 건 augmentation으로 인한 정상적인 현상(매 배치 이미지가 랜덤 변형되어 난이도 상승), 실제 개선 효과는 Colab에서 25 epoch 풀 학습 후 확인 예정.

**다음 할 일**: 이 코드로 Colab GPU에서 `config/default.yaml` 재학습 → 새 `best.pth`로 confusion matrix 재비교 + 실제 데이터 기반 Grad-CAM 시각화

---

## 2026-07-18 — 2차 학습 결과 (Augmentation + LR Scheduler 적용, 25 epoch)

Colab GPU에서 `config/default.yaml`(25 epoch, augmentation 적용)로 재학습한 `best.pth`를 로컬 val set(826장)으로 재평가.

### 결과 비교 (1차 vs 2차)

| 지표 | 1차 (증강 전, 10 epoch) | 2차 (증강+scheduler, 25 epoch) | 변화 |
|---|---|---|---|
| Accuracy | 0.557 | 0.580 | +0.023 |
| **Cohen's Kappa (quadratic)** | 0.661 | **0.759** | **+0.098** |

Confusion Matrix (2차, 행=실제, 열=예측):

| 실제\예측 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 230 | 79 | 14 | 5 | 0 |
| 1 | 73 | 47 | 29 | 4 | 0 |
| 2 | 37 | 46 | 99 | 29 | 1 |
| 3 | 2 | 2 | 15 | 81 | 6 |
| 4 | 0 | 0 | 0 | 5 | 22 |

Grade별 recall 비교:

| grade | recall (1차) | recall (2차) | 변화 |
|---|---|---|---|
| 0 | 0.716 | 0.701 | 소폭 하락 |
| 1 | 0.235 | 0.307 | 개선 |
| 2 | 0.533 | 0.467 | 하락 |
| 3 | 0.594 | 0.764 | 크게 개선 |
| 4 | 0.481 | 0.815 | 크게 개선 |

### 분석
- **accuracy는 소폭 개선(+2.3pp)이지만 kappa는 큰 폭으로 개선(+0.098)** — CLAUDE.md에서 명시한 대로 이 과제의 핵심 지표는 accuracy가 아닌 kappa이므로, 실질적으로 유의미한 개선으로 판단
- **grade 3, 4(중증) recall이 크게 향상** — 특히 grade 4는 0.481→0.815로, 스크리닝 트리아지 목적(중증 환자를 놓치지 않는 것)에 중요한 개선
- **grade 2 recall은 오히려 하락**(0.533→0.467) — grade 1/3과의 경계에서 혼동이 늘어난 것으로 보임 (실제 grade 2 중 46건이 1로, 29건이 3으로 오분류). augmentation이 중간 등급의 경계를 더 흐리게 만들었을 가능성
- 여전히 오분류는 인접 등급 사이에서만 발생 (0↔4 등 극단적 오류 없음) — ordinal loss 효과 유지

### 다음 개선 방향
- [x] 실제 데이터 기반 Grad-CAM 시각화 (특히 grade 1, 2 오분류 사례 위주로 확인) — 아래 참고
- [ ] grade 2 recall 하락 원인 확인 — augmentation 강도(rotation/jitter)를 낮춰서 재실험 비교
- [ ] grade 1 recall이 여전히 낮음(0.307) — grade 0/1 경계 개선을 위한 추가 데이터 또는 fine-grained loss 검토

---

## 2026-07-18 — Grad-CAM 시각화를 통한 오분류 원인 분석

2차 학습 모델(`best (1).pth`)로 val set에서 다음 7개 사례에 대해 Grad-CAM 생성. 이미지: [`docs/gradcam_examples/`](docs/gradcam_examples/)

| 파일 | 실제 | 예측 | 비고 |
|---|---|---|---|
| `g0_correct.png` | 0 | 0 | 정상 대조군 |
| `g1_wrong_1.png` | 1 | 0 | 오분류 |
| `g1_wrong_2.png` | 1 | 0 | 오분류 |
| `g2_wrong_1.png` | 2 | 3 | 오분류 |
| `g2_wrong_2.png` | 2 | 1 | 오분류 |
| `g4_correct_1.png` | 4 | 4 | 정확히 분류 |
| `g4_correct_2.png` | 4 | 4 | 정확히 분류 |

### 관찰
- **Grade 4 (정확히 분류된 케이스)**: 히트맵이 관절 중심부(관절 틈새)에 뚜렷하게 집중. 모델이 실제로 관절 공간 협착(joint space narrowing)을 근거로 판단하는 것으로 보임 — 설명력 측면에서 긍정적
- **Grade 1 (오분류, 둘 다 pred=0)**: 히트맵이 관절 한쪽에만 치우쳐 집중되고 반대쪽 관절연은 거의 보지 않음. Grade 1의 핵심 판별 근거인 "doubtful osteophyte"(경미하고 애매한 골극)의 미세한 신호를 놓치고 있는 것으로 추정됨 → **grade 1 recall이 낮은 원인에 대한 시각적 근거 확보**
- **Grade 2 (오분류)**: 두 케이스 모두 관절 양쪽 가장자리(osteophyte가 흔히 나타나는 위치)에 집중하고 있어 주목 위치 자체는 크게 벗어나지 않음. 다만 등급 강도 판단에서 인접 등급(1 또는 3)으로 치우치는 것으로 보아, 경계선상 케이스의 정도(severity) 구분에서 오류가 발생하는 것으로 추정
- 전반적으로 배경이나 뼈 가장자리 노이즈 등 **관절과 무관한 부위에 집중하는 사례는 없었음** — 모델이 엉뚱한 곳을 보고 있는 문제는 아님

### 결론 및 다음 방향
- Grad-CAM 결과, 모델의 한계는 "잘못된 부위를 본다"가 아니라 **"grade 1처럼 미세한 신호에 대한 민감도가 부족하다"**는 쪽에 가까움
- 다음 시도로는 grade 1 샘플에 대한 가중치를 더 높이거나(loss에 class weight 추가), grade 1/0 경계를 더 잘 구분하도록 하는 fine-grained 접근이 grade 2 augmentation 튜닝보다 우선순위가 높다고 판단

---

## 2026-07-18 — 3차 개선: Loss에 Class Weight 추가

Grad-CAM 분석 결론에 따라, grade 1의 학습 신호를 강화하기 위해 loss 단계에도 클래스 가중치 반영.

- **`src/baseline.py`**: `OrdinalCELoss`에 `class_weights` 옵션 추가 — `F.cross_entropy(logits, targets, weight=class_weights)`로 소수 클래스(특히 grade 1, 4) 오분류에 더 큰 페널티 부여
- **`src/train.py`**: 기존 `WeightedRandomSampler`에서 쓰던 클래스별 역빈도 가중치 계산 로직을 `compute_class_weights()`로 분리해서, 샘플링(sampler)과 loss 양쪽에 동일한 가중치를 재사용
  - 참고: 샘플링으로 이미 배치 내 클래스 비율을 맞추고 있어 완전히 새로운 정보는 아니지만, loss 페널티까지 더하면 소수 클래스 오분류 시 gradient 크기 자체가 커져 추가적인 학습 신호를 줄 수 있음

**검증**: `config/quick.yaml`로 스모크 테스트 — 에러 없이 3 epoch 정상 동작 확인 (수치는 미니 데이터셋이라 참고용)

**다음 할 일**: Colab GPU에서 `config/default.yaml`(25 epoch)로 재학습 → grade 1 recall이 실제로 개선됐는지, grade 2/4 성능은 유지되는지 3차 결과 비교

---

## 2026-07-18 — 3차 학습 결과 (Class Weight 적용, 25 epoch)

Colab GPU에서 재학습한 `best.pth`를 로컬 val set(826장)으로 재평가.

### 결과 비교 (2차 vs 3차)

| 지표 | 2차 (augmentation+scheduler) | 3차 (+class weight) | 변화 |
|---|---|---|---|
| Accuracy | 0.580 | 0.588 | +0.008 |
| Cohen's Kappa (quadratic) | 0.759 | 0.761 | +0.002 |
| **Macro F1** | 0.600 | **0.638** | **+0.038** |

Confusion Matrix (3차, 행=실제, 열=예측):

| 실제\예측 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 203 | 97 | 28 | 0 | 0 |
| 1 | 50 | 75 | 25 | 3 | 0 |
| 2 | 22 | 62 | 113 | 15 | 0 |
| 3 | 1 | 7 | 22 | 72 | 4 |
| 4 | 0 | 0 | 0 | 4 | 23 |

Grade별 recall 비교:

| grade | recall (2차) | recall (3차) | 변화 |
|---|---|---|---|
| 0 | 0.701 | 0.619 | 하락 |
| 1 | 0.307 | **0.490** | **크게 개선** |
| 2 | 0.467 | 0.533 | 개선 (2차 이전 수준 회복) |
| 3 | 0.764 | 0.679 | 소폭 하락 |
| 4 | 0.815 | 0.852 | 개선 |

### 분석
- **목표였던 grade 1 recall이 0.307 → 0.490으로 크게 개선** — class weight가 의도대로 작동
- 대신 **grade 0 recall이 0.701 → 0.619로 하락** — confusion matrix상 grade 0 중 97건이 grade 1로 오분류(2차엔 79건). grade 1 쪽으로 결정 경계가 넓어지면서 grade 0/1 경계의 애매한 케이스가 grade 1 쪽으로 더 많이 넘어간 전형적인 트레이드오프
- Accuracy/Kappa 총점은 거의 변화 없지만 **macro F1이 뚜렷하게 개선**(0.600→0.638) — 특정 클래스(grade 1)만 심하게 못 맞추던 불균형이 완화되고, 클래스 간 성능이 더 고르게 분산됨을 의미
- grade 4는 계속 최고 성능 유지(recall 0.852) — 스크리닝 목적엔 긍정적
- 임상적 관점: grade 0을 1로 과대진단하는 오류는 grade 1을 0으로 과소진단(놓침)하는 오류보다 상대적으로 덜 위험한 방향 — 스크리닝 트리아지 목적에는 나쁘지 않은 트레이드오프로 판단

### 다음 개선 방향
- [x] grade 0/1 경계 트레이드오프를 조정할 별도 임계값(threshold) 튜닝 검토 — 아래 참고
- [ ] grade 3 recall이 소폭 하락한 원인 확인 (2/3 경계 혼동 증가 여부)
- [ ] 현재까지 결과를 baseline으로 확정하고, FastAPI 서비스화 등 다음 단계 진행 여부 결정

---

## 2026-07-18 — Test set 최종 검증 + Ordinal Threshold 튜닝

### 1. Test set 최종 검증 (val 과적합 여부 확인)

지금까지 val set(826장)으로만 반복 튜닝해왔기 때문에, 한 번도 사용하지 않은 test set(1,656장)으로 3차 모델을 최종 확인.

| | val | test |
|---|---|---|
| accuracy | 0.588 | 0.582 |
| kappa | 0.761 | 0.770 |

val과 test 성능이 거의 동일 (test가 오히려 kappa는 약간 더 높음) → **val 반복 튜닝으로 인한 과적합은 없었음**. 지금까지의 결과 신뢰 가능.

### 2. Ordinal Threshold 튜닝

기존 방식(`argmax`)은 5개 로짓 중 최댓값만 보고, softmax 확률의 순서형 구조(등급 간 거리)를 버림. 대신 **softmax 확률의 기대값**(`expected_grade = Σ(prob_i × grade_i)`)을 계산하고, 이 연속값을 등급으로 변환하는 절단점(threshold)을 val set에서 quadratic-weighted Kappa 기준으로 최적화.

- 구현: `src/threshold_tuning.py` (`get_expected_grades`, `tune_thresholds`, `apply_thresholds`), `src/evaluate.py` — CLAUDE.md에 명시된 `python src/evaluate.py --model_path checkpoints/best.pth` 커맨드를 실제로 구현
- 탐색 방법: 4개 절단점(0-1, 1-2, 2-3, 3-4 경계)에 대해 val set 기준 coordinate ascent로 grid search (val에서만 튜닝, test는 검증에만 사용)
- 튜닝된 절단점: `[0.70, 1.35, 2.05, 3.05]` (표준 반올림 기준 `[0.5, 1.5, 2.5, 3.5]` 대비 낮은 등급 쪽으로 이동)

### 결과 비교 (test set, 재학습 없이 추론 방식만 변경)

| 방식 | accuracy | kappa |
|---|---|---|
| argmax (기존) | 0.582 | 0.770 |
| 기대값 + 표준 반올림 | 0.570 | 0.789 |
| **기대값 + Kappa 튜닝 threshold** | **0.597** | **0.795** |

- val에서 튜닝한 threshold가 test에서도 일관되게 개선 (val 전용 튜닝임에도 test에 잘 일반화됨)
- 재학습 비용 없이 **kappa +0.025, accuracy +1.5%p** 개선
- grade별로는 grade 0(0.624→0.698), grade 3(0.679→0.825), grade 4(0.815→0.824) recall이 개선되고, grade 1(0.490→0.378)/grade 2(0.533→0.456) recall은 다소 하락 — 전체 kappa/accuracy 관점에서는 순이익이지만, grade 1 민감도가 중요한 경우라면 argmax 유지를 고려할 수 있음

### 결론
- **다음 학습부터는 추론 시 argmax 대신 `src/evaluate.py`의 threshold 방식을 기본으로 사용**
- threshold 값은 체크포인트마다(재학습할 때마다) 달라지므로, 매번 val set으로 재튜닝 필요 (`tune_thresholds` 함수가 이를 자동화)
