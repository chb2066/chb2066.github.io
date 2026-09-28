---
title: 전통문양 감성 라벨 예측
summary: 전통문양 이미지에 감성 형용사 5개를 다는 태깅 모델.
context: AiRLab · ETRI 과제
period: 2026.06 ~ 2026.08
role: 실험 설계, 학습, 분석
stack: PyTorch, DINOv3, klue-roberta
tags: [Vision-Language, Multi-label]
date: 2026-08-26
draft: false
---

## Setup

과제는 전통문양 이미지에 감성 형용사(Ex:풍성한, 신비한, 고전적인)를 다는 태깅 모델을 만드는 것이다. 22개 어휘에 대한 multi-label classification 이고, 학습 데이터는 이미지마다 정답 형용사 5개가 주어지며 추론 데이터에 대해서도 예측 5개를 내야 한다. 이미지는 흑백 선화다. 정답과 예측이 모두 5개이므로 precision과 recall은 같고, 지표는 F1을 사용한다.

```
top-5 F1 = |예측 ∩ 정답| / 5
```

출력 라벨 수가 5개 고정이므로 라벨 선택은 threshold 없이 상위 5개를 취하는 top-k 로 한다. 데이터는 이미지와 메타 데이터 json으로 구성되며, json에는 이미지에 대한 설명인 `description`과 구조적 메타데이터 9개 필드가 함께 있다. `description`은 문화포털이 제공하는 이미지 설명 데이터이다. 학습 label인 22개 감성 라벨은 전문가가 이미지와 description을 보고 라벨링한 감성 형용사이다. 과제 요구사항은 이미지와 부속 정보를 함께 쓰되 이미지가 예측에 영향을 주는 것이다. 용역 요구 목표는 0.80이다.


## 이미지 실험 결과
| 접근 | Model | F1@5 |
|---|---|---|
| CNN model | ConvNeXtV2-large | 0.56 |
| ViT model | EVA-02-large, end-to-end | 0.571 |
| CLIP 계열 model + cls head | SigLIP2 | 0.567 |
| CLIP 계열 model + cls head | FG-CLIP2 | 0.576 |
| foundation model + cls head | DINOv3 | 0.579 |


## 이미지 + description 실험 결과
| 접근 | Model | F1@5 |
|---|---|---|
| CLIP 계열 model + cls head | SigLIP2 | 0.622 |
| CLIP 계열 model + cls head | FG-CLIP2 | 0.623 |


'이미지 단독', '이미지 + description' 두 축으로 실험을 진행하였다. 실험 결과, 이미지만 단독으로 사용하는 것보다, 이미지와 description 정보를 동시에 사용하는 것이 성능적 우위를 보였다.
하지만, 실험 결과 내 최대성능인 FG-CLIP2 + cls head 의 구성도 용역 요구 성능인 0.8에는 한참 못미치기에 부가적인 방법 고안이 필요했다. 실험에서 알 수 있었던 건 크게 두 가지였는데, 하나는 이미지 자체로는 성능 상한이 명확히 존재한다는 점, 나머지 하나는 description 정보와 같은 부가 메타데이터가 성능에 영향을 끼친다는 점이었다. 따라서 이후 방향은 description과 같은 부가 메타 데이터들의 성능 영향을 평가하고 해당 데이터들을 최대한 활용할 방법을 찾는 방향으로 실험을 진행했다.

## 입력 데이터 성능 분석



| 채널 | 방법 | F1@5 |
|---|---|---|
| Description | klue/roberta-large, fine-tune | 0.774 |
| 메타데이터 9개 필드 | multi-hot → linear probe | 0.561 |
| 이미지 | DINOv3 | 0.579 |


같은 split에서 각 채널을 단독으로 측정했다. 실험 결과, bert 계열 모델을 사용하여 Description 데이터를 단독으로 사용할 때 가장 높은 성능을 보였다. Description을 제외한 다른 메타 데이터들은 전반적으로 성능에 끼치는 영향이 적었으며, 이후 단일 채널 기준으로 여러 모델을 실험해봤으나, Description을 제외한 데이터의 성능 상한은 0.6 이하에 그쳤다.

성능의 차이는 두 채널에서 뽑아낼 수 있는 특성량의 상한이 다르다는 것과 description 내 라벨이 포함된 정도에 기반한 차이라고 추론했다. 이미지는 흑백 선화라 색과 질감이 없고, 시각적으로 거의 같은데 라벨이 다른 문양이 많다. description은 서술량이 길고 형용사 키워드의 의미가 내포된 경우가 많다. 그래서 설계 문제는 적은 데이터에서 텍스트를 최대로 쓰면서 이미지를 동시에 반영할 방법을 고안하는 것을 주요 목적으로 했다.


## 텍스트 데이터 성능 실험

| 채널 | 방법 | F1@5 |
|---|---|---|
| Description | klue/roberta-large, fine-tune | 0.774 |
| FG-CLIP2 | text encoder only, fine-tune | 0.561 |
| FG-CLIP2 | frozen image encoder + unfrozen text encoder only | 0.621 |


실제로 텍스트 데이터가 성능에 주는 영향을 파악하기 위해, 이전 실험에서 나온 bert 계열 모델을 대조군으로 기존 최고 성능을 낸 FG-CLIP2를 사용하여 실험을 진행했다. 실험 결과, FG-CLIP2에서는 텍스트만을 사용하여 학습을 진행하더라도, bert 계열보다 낮은 성능을 보이는 것을 확인하였다. 이를 통해 기존에 예상했던 텍스트와 이미지가 가진 특성량의 상한이 다르다는 것이 주된 차이가 아니라, 모델 구조에 의한 텍스트 예측 방식이 중요하다는 것을 알게 됐다. FG-CLIP2에서는 해당 인코더를 사용하고 별도로 cls head를 달아서 학습하기에 임베딩과 classification이 별개로 작동하였다. 하지만, bert 계열 같은 경우 cls token을 넣고 masked token prediction 하는 방식이기에 기존 head를 달아서 하는 방식과 달리, 임베딩과 cls가 한 과정 내에 작동한다. 따라서, 감성 형용사라는 라벨이 문장 내 성분과 같이 작동할 수 있다는 점과 bert 자체의 masked token을 predict하는 구조가 성능에 큰 영향을 미친 것으로 유추한다.

## 구조 설계
위 특징들을 토대로, bert 계열 모델들을 사용하면서 이미지의 feature가 출력 label에 영향을 미칠 수 있는 형태로 설계를 진행했다.


```
이미지 ─▶ DINOv3 (frozen) ──────▶ proj ─▶ 1 token ─┐
                                                  ├─▶ self-attn ─▶ linear ─▶ 22
텍스트 ─▶ 한국어 인코더 (fine-tune) ▶ proj ─▶ 1 token ─┘
```

Dinov3로 추출한 이미지의 특징과 bert 모델로 추출한 text 특징을 attention으로 aggregation 하는 방식을 채용하였다. 이 방식을 통해 이미지의 특징과 description의 특징을 상보적으로 사용하되, text encoder model은 train data에 finetune하고 lr을 낮게 사용함으로써, 이미지와 텍스트의 특징 반영량을 다르게 하면서 bert 계열 모델의 특징은 조금씩 반영시키는 방식을 구성했다.


### 최종 시스템

텍스트 인코더 다섯 개를 각각 동일한 frozen DINOv3 feature와 융합한 뒤 앙상블했다.

| 텍스트 인코더 | F1@5 |
|---|---|
| kobigbird-bert-base | 0.7861 |
| klue/roberta-large | 0.7794 |
| xlm-roberta-large | 0.7753 |
| bert-base-multilingual-cased | 0.7753 |
| kcbert-large | 0.7682 |

후보 6개의 부분집합 63개를 전부, Dirichlet 가중 랜덤서치로 탐색했다. 인코더 4개 0.7991, 6개 전부 0.7996, 그리고 최선의 5개(klue + xlmr + kcbert + mbert + kobigbird)가 0.8000이었다.

### 오차

- 다섯 개를 전부 놓친 샘플은 한 장도 없고, 77.1%가 최소 네 개를 맞혔다.
- 최고: 풍성한 0.909, 단순한 0.893, 귀여운 0.881. 최저: 신비한 0.656, 고급스러운 0.660, 현대적인 0.691.
- 어려움은 희소성이 아니라 추상성을 따라간다 - 가장 드문 라벨인 익살스러운이 0.824인데, 훨씬 흔한 우아한은 이보다 예측 성공률이 낮다.
- 앙상블 다섯은 Jaccard 0.74~0.79로 일치하며, 무관한 예측을 하는 게 아니라 애매한 경계 라벨에서 성능적 향상을 일으켰다.

<!--MINE-->
예측을 들여다보면, 모델이 학습한 것의 일부는 개별 라벨이 아니라 라벨 집합이다. 집합 안에서는 희소한 라벨도 되찾아내고, 추론된 집합 내 포함되지 않는 라벨은 아무리 흔해도 덜 나온다. 이는 남은 오차를 모델이 아니라 라벨 정의와 관련됨을 시사한다.
<!--/MINE-->

---

모든 수치는 seed 42로 고정한 하나의 validation split에서 나왔고, bf16 비결정성 아래 ±0.2점 안에서 재현된다. 데이터셋 구성과 레코드별 세부사항은 과제 자료라 생략했다.
