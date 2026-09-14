<!--
개정: 2026-09-10 (원본: src/content/reviews/dinov2.md)
- 원본 마지막 문장이 "먼저 학습시켜 고정한 뒤 distill" 에서 끊겨 있어 논문 5절로 완성
- 데이터 구축에 실제 수치 보강 — uncurated 1.2B → LVD-142M, Faiss, k-means 기반
  retrieval 규칙(query pool 크기에 따라 N개 최근접 / 클러스터에서 M개 샘플링)
- 「효율화」 신설 — FlashAttention 자체 구현, sequence packing, efficient stochastic depth, FSDP.
  논문 5절. 이게 "대규모로 밀어붙일 수 있었던 이유"라 원본의 서술과 직접 이어진다
- 고해상도 단계의 구체값(518×518, 사전학습 말미 짧은 기간) 확인해 반영
- distillation 의 세부 조건(frozen teacher, student EMA를 최종 모델로, masking·stochastic depth
  제거, global crop 2개에만 iBOT loss) 보강
- 「관련 연구」를 「배경 지식」으로 재구성하고 끝맺음을 평서형으로 통일
- 원본의 iBOT 대비 차이 5가지 정리는 그대로 유지. 이 글의 핵심이다
- 2차 개정 (사용자가 직접 고친 OpenVLA 수정본의 편집 방향을 반영):
  핵심 키워드 / 사용 가능 분야 블록 삭제, 주요 전략을 「#### 주요 전략」 번호 목록으로 교체,
  가운뎃점을 쉼표로, 메타 코멘트와 구어체 소제목 정리. 강조 볼드는 사용자 요청으로 유지
- 3차 개정: 학습 설정(하이퍼파라미터 나열) 블록 제거, 구어체 소제목을 명사구로,
  굵은 소제목 뒤에 빈 줄을 넣어 본문과 같은 줄로 붙던 렌더링 문제 수정
- 4차 개정: 닫는 ** 앞이 따옴표/괄호면 볼드가 적용되지 않던 문제 수정
  (구두점을 볼드 밖으로). 목록 항목의 종결을 음슴체로 통일
-->
---
title: DINOv2
paper: "DINOv2: Learning Robust Visual Features without Supervision"
venue: TMLR 2023
link: https://arxiv.org/abs/2304.07193
claim: 정제된 데이터로 retrieval 큐레이션한 대규모 데이터셋과 iBOT 기반 목적함수로, 파인튜닝 없이 쓸 수 있는 범용 시각 특징을 학습한다.
tags: [Self-supervised, Vision Backbone]
tier: main
date: 2025-06-10
draft: false
---

#### 주요 전략
1. 정제된 데이터를 검색 query로 삼아 비정제 데이터에서 유사한 것만 retrieval하는 방식으로 자동 큐레이션함.
2. DINO와 iBOT loss를 함께 쓰고, 대규모 ViT에서 소형 모델로 distillation함.

#### 배경 지식

**Self-supervised learning의 두 갈래**

- **Intra-image SSL** — 이미지의 가려진 부분을 복원하는 방식임. 표현은 나오지만 fine-tuning을 해야 쓸 수 있음.
- **Discriminative SSL** — 이미지 그룹 간 판별 신호로 특징을 학습함. DINO, iBOT이 여기 속함.

**Scaling의 문제**

데이터 양을 늘려 성능을 올린 연구들이 있었지만 **데이터 품질이 낮다**는 게 걸림돌이었다. 웹에서 긁어온 이미지는 소수의 지배적인 모드에 쏠려 있어서, 양이 늘어도 다양성이 따라오지 않는다.

**Image Retrieval을 큐레이션에 쓰기**

ImageNet 같은 정제된 데이터로 특징을 뽑고, 인터넷의 비정제 데이터로도 특징을 뽑은 뒤, retrieval로 visual similarity를 재서 **좋은 데이터와 분포가 비슷한 것만 학습 데이터에 추가**하는 방식이다.

검색 규칙은 query pool 크기에 따라 달라진다.

- 기준 데이터가 크면 → 각 기준 이미지마다 가장 가까운 **N개**를 찾아옴.
- 기준 데이터가 작으면 → 기준 이미지가 속한 **클러스터에서 M개**를 샘플링함. 소수 데이터를 많이 가져온다는 뜻임.

유사도는 `q·k` 내적으로 구한다.

#### 데이터 구축 — LVD-142M

비정제 데이터 풀에서 정제 데이터와 닮은 이미지를 retrieval해서 최종 데이터셋을 만든다.

1. 웹에서 `img` 태그의 URL을 추출함. 부적절 URL 필터링, NSFW 필터링, 얼굴 블러 처리를 거쳐 **1.2B개의 고유 이미지**가 남음.
2. 중복을 제거함. copy detection 파이프라인을 써서 비정제 데이터 안의 중복을 먼저 없애고, 정제 데이터와 겹치는 것도 뺌.
3. ViT 모델로 임베딩하고 이미지 간 유사도는 **cosine similarity**로 측정함.
4. 비정제 데이터에 **k-means clustering**을 적용한 뒤 위의 retrieval 규칙으로 뽑음.

결과가 **LVD-142M**, 1억 4200만 장이다. 검색과 중복 제거는 Faiss 라이브러리로 처리한다.

여기서 눈여겨볼 점은 **정제 데이터가 라벨이 아니라 query로 쓰인다**는 것이다. 라벨을 쓰지 않으면서도 "무엇이 좋은 데이터인가"의 기준을 정제 데이터가 대신 정해준다.

#### 기존 방법과의 차이

**iBOT loss와 다른 다섯 가지**

**1. 짧은 고해상도 단계**

낮은 해상도로 학습하다가 **사전학습이 끝나기 직전 짧은 기간 동안만 518×518로 올린다.** 분할이나 검출처럼 픽셀 수준 태스크에서는 해상도가 중요한데 — 작은 물체가 저해상도에서 사라진다 — 처음부터 고해상도로 학습하면 시간과 메모리가 너무 든다. 그래서 말미에만 짧게 붙인다.

**2. head 분리**

DINO loss와 iBOT loss가 각각 학습 가능한 MLP projection head를 쓴다. 기존에는 두 loss가 head를 공유하는 게 낫다고 알려져 있었는데, **규모를 키우면 반대가 참**이라는 것을 관찰했다. 그래서 class token용과 patch token용 **MLP head를 2개 따로** 둬서 각 특징을 살린다. 서로 영향을 주지 않고 독립적으로 작동한다.

**3. teacher 확률 분포의 생성 방식**

teacher가 feature를 뽑고 projection head로 점수를 매기는 데까지는 같다. 그 다음이 갈린다.

- **기존(DINO)** — softmax로 확률을 출력한 뒤 정답의 이동평균을 저장함. centering으로 특정 칼럼이 정답으로 많이 나왔으면 점수를 깎고, sharpening으로 temperature를 조절함.
- **DINOv2** — MLP를 통과시켜 logit을 뽑고, 배치 내 전체 이미지가 정답 칸에 골고루 들어가도록 **SwAV의 Sinkhorn-Knopp 정규화**를 씀. 모든 칼럼을 각 칼럼 내 logit 합으로 나눠서, 특정 칼럼이 크면 패널티를 주고 작으면 어드밴티지를 주는 구조임. Sinkhorn-Knopp은 3회 반복함.

**4. KoLeo regularizer 추가**

각 특징 벡터들이 개별적으로 구별되도록 만든다. 최근접 이웃까지의 거리가 0이 되지 않게 log 거리 값을 최대화한다. 표현이 한 점으로 뭉치는 것을 막는 장치다.

**5. teacher를 momentum encoder로 두지 않음**

가장 큰 모델인 **ViT-g(1B 파라미터)를 먼저 학습시켜 고정한 뒤, 거기서 작은 모델들로 distill한다.** 처음부터 학습시키는 대신 큰 모델을 교사로 쓰는 것이다.

distillation의 세부 조건은 이렇다. 목적함수 자체가 이미 teacher에서 student로의 distillation 형태이므로 **같은 학습 루프를 쓰되 몇 가지만 바꾼다.**

- 더 큰 모델을 **frozen teacher**로 씀.
- student의 EMA를 따로 유지해서 **그것을 최종 모델로 삼음.**
- masking과 stochastic depth를 **제거**함.
- iBOT loss는 **두 개의 global crop에만** 적용함.

ablation에서 이 방식이 처음부터 학습시키는 것보다 낫다는 것이 확인됐고, **ViT-L 크기에서도 그랬다.**

#### 효율화

대규모로 밀어붙일 수 있었던 이유가 여기 있다. A100 GPU에서 PyTorch 2.0으로 학습한다.

- **FlashAttention 자체 구현** — 메모리 효율적인 attention을 직접 구현했음.
- **Sequence packing** — DINO 알고리즘은 224 해상도의 large crop과 그보다 작은 local crop을 함께 forward해야 함. 길이가 다른 시퀀스를 따로 돌리면 낭비가 크므로, NLP에서 온 sequence packing으로 **여러 시퀀스를 이어붙여 한 번에 처리**함.
- **Efficient stochastic depth** — 건너뛸 층의 연산을 실제로 생략하도록 개선했음.
- **FSDP (Fully-Sharded Data Parallel)** — AdamW의 옵티마이저 상태를 여러 GPU에 분산함. 그래서 **모델 크기가 한 GPU 메모리에 묶이지 않음.** GPU 간 통신 비용도 줄어듦.

#### 정리

이 논문에서 가져갈 만한 것은 두 가지다.

**첫째, 정제 데이터를 라벨이 아니라 검색 query로 쓰는 발상.** "무엇이 좋은 데이터인가"를 사람이 라벨링하는 대신, 이미 좋다고 알려진 데이터와의 유사도로 정의한다. 라벨 없이 큐레이션이 가능해진다.

**둘째, 규모가 바뀌면 최적 설계가 뒤집힌다는 관찰.** head 공유가 작은 규모에서는 낫지만 큰 규모에서는 분리가 낫다는 것을 실제로 확인했다. 작은 실험에서 얻은 설계 결정을 그대로 스케일업하면 안 된다는 뜻이다.
