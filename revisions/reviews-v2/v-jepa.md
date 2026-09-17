---
title: V-JEPA
paper: Revisiting Feature Prediction for Learning Visual Representations from Video
venue: TMLR 2024
link: https://arxiv.org/abs/2404.08471
claim: 비디오에서도 픽셀이 아니라 임베딩 공간에서 마스킹된 시공간 영역을 예측하게 하면 강한 시각 표현이 학습된다.
tags: [Self-supervised, Video, JEPA]
tier: basic
date: 2025-07-16
draft: true
---

> 📄 [**Revisiting Feature Prediction for Learning Visual Representations from Video**](https://arxiv.org/abs/2404.08471) · TMLR 2024 · Bardes, Garrido, Ponce et al.

## Abstract

**문제**
비디오 SSL의 주류는 **마스킹된 픽셀(복셀)을 복원**하는 방식이었음
복원 목표는 저수준 디테일에 용량을 쓰고, 얻어진 표현은 **얼린 채로는 쓰기 어려움**

**해결책**
복원 대신 **EMA target encoder가 만든 잠재 표현**을 예측 목표로 삼음
시간축 전체에 같은 공간 마스크를 반복 적용하는 **multi-block masking** 으로 시간적 지름길을 막음
얼린 백본 하나로 동작 중심 태스크와 외형 중심 태스크를 모두 커버

---

## 1. Introduction

![Figure 1](/img/v-jepa/f1.png)

**출발 질문** - 시각 세계를 인식하는 능력을 어떻게 학습시킬 것인가. 이 논문은 정적 이미지가 아니라 **비디오로부터 시각적 잠재 표현을 학습**하는 쪽을 택한다.

**기존 접근 ① 텍스트 캡션을 쓰는 방법**
- CLIP 계열 - 웹에서 긁은 대규모 이미지·텍스트 쌍으로 학습. **백본을 얼린 채 가벼운 head만 붙여도** 되므로 end-to-end 학습이 불필요
- 비디오 확장 - VideoBERT는 캡션을 써서 마스킹 영역의 표현을 예측하게 했고, VideoCLIP은 비디오 캡션의 잠재 표현으로 contrastive learning 수행. MERLOT, VATT, InternVideo가 이 흐름을 확장
- 정리하면 **캡션으로 잠재 공간이나 마스크 영역을 예측하게 하는 방식이 많았음**

**기존 접근 ② 라벨 없는 SSL**
- **불변성 기반** - 손으로 만든 이미지 변형에 대해 불변인 표현을 학습함 → **태스크 특화 귀납 편향을 많이 요구**해서 적용 범위가 제한됨
- **재구성 기반** - DAE는 손상된 입력의 복원으로 표현을 학습할 수 있음을 보였고, MAE는 마스킹된 패치의 **픽셀**을 예측하게 함
- MAE는 파인튜닝하면 좋지만, **얼린 상태의 표현은 불변성 기반에 미치지 못함** → 방향이 **날 픽셀보다 잠재 공간에서 예측하는 쪽**으로 잡힘
- DINOv2는 불변성 loss와 masked image modeling loss를 결합해 경쟁력을 보였으나 **정적 이미지에 머물러 동적 정보 수집은 기대하기 어려움**

**기존 접근 ③ 비디오에서의 SSL**
- 시간에 걸쳐 불변인 잠재 공간 학습, 다음 프레임 표현의 계층적 학습, segmentation 활용, **마스킹된 시공간 복셀**을 예측하는 encoder-decoder 확장 등
- 사전학습된 CLIP encoder의 **얼린 잠재 공간에서** 마스크 모델링 loss를 계산한 사례도 있음
- **V-JEPA는 얼린 외부 인코더를 쓰지 않고, 학습 중에 온라인으로 잠재 공간을 예측**한다는 점이 다름

**본 논문의 기여**
1. 비디오에서 **feature prediction 자체만으로** 경쟁력 있는 표현이 나온다는 것을 보임
1. 픽셀 예측 대비, 얼린 평가(attentive probing)에서 일관되게 앞섬을 검증
1. 마스킹 전략·데이터 분포·평가 프로토콜 각각이 얼마나 기여하는지 분해

---

## 3. Methodology: Video-JEPA

![Figure 2](/img/v-jepa/f2.png)

**JEPA의 기본 발상** - 한 곳을 보고 다른 곳을 예측한다. 네트워크 세 개를 쓴다.

| | context encoder | target encoder | predictor |
|---|---|---|---|
| 입력 | 마스킹되고 남은 패치 | 원본 비디오 전체 | context token + mask token |
| 보는 것 | 비디오의 일부 | 비디오 전체 | - |
| 갱신 | loss로 학습 | **EMA로 갱신** | loss로 학습 |
| 출력 | 보이는 부분의 context representation | 정답 벡터 | 맥락 기반 예측 |

세 네트워크의 목적은 **predictor의 출력과 target encoder의 출력 사이의 손실을 최소화**하는 것이다.

### 3.1 Training Objective

![Figure 3](/img/v-jepa/f3.png)

**학습 절차**
1. 비디오 클립을 마스킹해 context encoder에 넣어 context representation을 만듦 → **남아 있는 패치의 잠재 표현이 타겟 영역을 예측하도록** 만드는 것
1. predictor가 context token을 받아 사라진 영역을 예측함. **학습 가능한 mask token**을 함께 넣는데, mask token은 **공유된 학습 가능 벡터와 3D sin-cos positional embedding의 합**
1. predictor의 예측과 target encoder의 출력을 **L1 loss** 로 비교

**Multi-Mask Prediction** - 같은 비디오에 대해 **0.15짜리 마스크를 8번, 0.7짜리 마스크를 2번** 독립적으로 예측하고, 오차들의 평균으로 loss를 구성한다.

**붕괴 방지** - **stop-gradient**. target encoder로는 gradient가 흐르지 않고 EMA로만 갱신된다. 이 비대칭이 표현이 한 점으로 무너지는 것을 막는다.

한 줄로 요약하면, **predictor가 가려진 부분을 예측하고, target encoder가 전체 영상을 정답지로 만들며, L1 loss로 둘의 차이를 줄인다. EMA가 정답지 기준을 안정화한다.**

### 3.2 Prediction Task: Predicting y from x

**Multi-block masking**
- **Short-range** - 8개 블록의 합집합이 프레임 면적의 약 **15%** 를 덮음
- **Long-range** - 2개 블록의 합집합이 약 **70%** 를 덮음
- 종횡비는 0.75~1.5 에서 랜덤 샘플링
- **공간 마스크를 시간축 전체에 그대로 반복 적용** → temporal mask ratio 가 사실상 1.0
- 최종 마스킹 비율 **약 90%**

시간축 전체에 같은 공간 마스크를 적용하는 것이 핵심이다. 그렇게 하지 않으면 다른 프레임에서 답을 그대로 볼 수 있어 과제가 너무 쉬워진다.

### 3.3 Network Parameterization

**입력 요소**
- 비디오에서 랜덤 시작점으로 **연속 64프레임 클립**을 뽑고, temporal stride 4로 **16프레임을 균등 샘플링**
- 16프레임 클립에 **3D conv**(2×16×16 필터 d개)를 적용해 `8×14×14×d` 텐서 생성
- **3D positional embedding**을 더하고 flatten 해서 `1568×d` 토큰 시퀀스로 만듦
- encoder 는 ViT, predictor 는 좁고 얕은 ViT

---

## 4. What Matters for Learning Representations from Video

### 4.1 Predicting Representations versus Pixels

![Table 1](/img/v-jepa/t1.png)

- 다른 조건을 모두 고정하고 **loss를 계산하는 공간만** 픽셀 ↔ 특징으로 바꿔 비교
- 특징 공간 예측이 **얼린 평가에서 일관되게 앞섬**
- 앞서 배경에서 짚은 "MAE의 표현은 얼린 상태로는 약하다"는 문제를 정면으로 겨냥한 실험

### 4.2 Pretraining Data Distribution

- 사전학습 데이터 분포를 바꿔가며 학습 → **동작 중심 데이터(SSv2)를 섞으면 동작 태스크가, 외형 중심 데이터(K710)를 섞으면 외형 태스크가 좋아짐**
- 둘을 섞은 분포가 양쪽을 동시에 커버

### 4.3 Evaluation: Attentive Probing

- 얼린 encoder 출력의 pooling 방식을 평균 풀링 ↔ **attentive pooling** 으로 비교
- attentive probing 이 얼린 표현의 품질을 더 정확히 드러냄 → 이후 모든 비교의 기본 프로토콜로 채택

### 4.4 Prediction Task: Predicting y from x

- 마스킹 전략 ablation. **시간축으로 마스크를 반복하지 않으면 성능이 떨어짐** → 시간적 지름길이 실제로 존재함을 확인

---

## 5. Comparison with Prior Work

![Table 6](/img/v-jepa/t6.png)

가장 큰 모델인 **ViT-H/16을 비디오만으로 학습**한 결과다. 전부 **파라미터를 전혀 조정하지 않은 frozen backbone** 기준이다.

| 태스크 | 성능 |
|---|---|
| Kinetics-400 (외형 기반) | **81.9%** |
| Something-Something-v2 (동작 기반) | **72.2%** |
| ImageNet-1K | **77.9%** |

이 표의 요지는 숫자 자체가 아니라 **같은 얼린 백본 하나로 성격이 다른 두 태스크를 모두 처리했다**는 데 있다. Something-Something-v2 는 시간적 동작을 이해해야 풀리고, Kinetics-400 은 외형만으로도 상당 부분 풀린다. 보통은 한쪽에 맞추면 다른 쪽이 약해진다.

- **5.1** 픽셀 예측 방식(OmniMAE, VideoMAE, Hiera)과의 직접 비교 - 얼린 평가에서 앞섬
- **5.2** 최신 모델과의 비교 - 텍스트를 쓰지 않고도 경쟁력 확보
- **5.3 Label-efficiency** - 라벨을 적게 쓰는 저샷 설정에서 격차가 더 커짐

---

## 6. Evaluating the Predictor

![Figure 6](/img/v-jepa/f6.png)

- predictor 의 출력을 시각화해서 **예측이 실제로 가려진 영역에 근거를 두고 있는지** 확인
- 예측이 **시공간적으로 일관**되며 원 비디오의 해당 영역과 대응됨 → 잠재 공간 예측이 임의의 벡터를 맞히는 것이 아님을 보임

---

## 정리

이 논문의 핵심은 "**무엇을 예측 목표로 삼을 것인가**"다.

픽셀을 목표로 삼으면 저수준 디테일을 맞히는 데 용량이 쓰이고, 얻어진 표현은 그대로 쓰기 어렵다. 잠재 표현을 목표로 삼으면 **무엇을 버릴지를 target encoder가 알아서 정한다.** 그리고 그 target encoder는 EMA로 천천히 따라오는 자기 자신이라, 외부 인코더에 의존하지 않는다.

I-JEPA가 이미지에서 한 일을 비디오로 옮긴 것이고, 옮기면서 추가된 설계는 **시간축 전체에 같은 공간 마스크를 적용하는 것**이다.

---

*그림은 모두 원 논문에서 가져왔다. Bardes et al., [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471), TMLR 2024.*
