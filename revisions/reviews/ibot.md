<!--
개정: 2026-09-10 (원본: src/content/reviews/ibot.md)
- 「정리」로 시작하던 구조를 i-jepa 형식(핵심 키워드/주요 전략/사용 가능 분야 → 배경 지식 →
  method → 세부 구조)으로 재배치
- 원본에서 문장이 뒤엉킨 곳 수정: "양쪽에 대해 동일하게 계산 후 평균이 loss는 두 개의
  글로벌 뷰에만 적용" → 두 문장으로 분리
- blockwise masking 의 prediction ratio 실제 값 보강 (r=0.3, 확률 0.5로 r=0 도 섞음)
- 결과 보강 — ImageNet linear probing 82.3%
- 「왜 시각 토크나이저가 어려운가」 신설. 언어와 달리 시각 의미는 단어 빈도 통계처럼
  자연스럽게 나오지 않는다는 논문 서론의 논지. 원본의 "pretrained VAE 단점" 서술과 이어진다
- iBOT 이 관찰한 창발 현상(local semantic pattern) 추가
- 끝맺음을 평서형으로 통일
- 원본의 DINOv2 와의 head 공유/분리 대조는 그대로 유지. 두 글이 이어지는 좋은 지점이다
-->
---
title: iBOT
paper: "iBOT: Image BERT Pre-Training with Online Tokenizer"
venue: ICLR 2022
link: https://arxiv.org/abs/2111.07832
claim: 별도로 사전학습한 고정 토크나이저 없이, EMA teacher를 온라인 토크나이저로 삼아 MIM과 self-distillation을 함께 최적화한다.
tags: [Self-supervised, Masked Image Modeling]
tier: main
date: 2025-05-14
draft: false
---

핵심 키워드:
Masked Image Modeling, online tokenizer, self-distillation, blockwise masking
주요 전략:
EMA teacher가 토크나이저 역할을 겸해서, 토크나이저를 미리 학습시키는 단계 자체를 없앤다
사용 가능 분야:
linear probing 분류, 객체 검출, instance/semantic segmentation

#### 요약

iBOT은 크게 두 가지 아이디어를 쓴다.

**1. DINO의 loss**
teacher와 student가 각각 다른 뷰를 보게 하고, student가 teacher의 확률 분포를 따라가게 만든다. EMA로 teacher의 업데이트량을 조절한다. teacher는 큰 뷰를, student는 작은 뷰를 보게 해서 부분에서 전체를 맞히도록 학습시킨다.

**2. iBOT의 MIM**
teacher는 마스킹하지 않은 원본 이미지(global view만)를 보고, student는 마스킹된 이미지를 본다. student가 보는 뷰는 global과 local을 섞는데 local이 훨씬 많다. 대략 2:8에서 2:10 정도다. **global을 넣는 이유는 그게 없으면 학습이 안 되기 때문이다.** 부분과 전체의 관계를 볼 수 없다.

기존 DINO가 cls 토큰만 썼다면, iBOT은 **patch loss를 추가**해서 마스킹된 패치를 맞히게 한다. 다만 DINO의 cls loss는 다른 뷰끼리 비교하는 반면, MIM은 마스킹된 패치를 다뤄야 하므로 **오직 두 개의 global view에만 적용**한다.

#### 배경 지식

**왜 시각 토크나이저가 어려운가**

BERT류의 성공은 언어를 **의미 있는 단위로 토큰화**할 수 있다는 데 기대고 있다. WordPiece 같은 것이다. 그런데 언어의 의미 단위는 단어 빈도 통계에서 자연스럽게 나오는 반면, **이미지는 연속적이라 시각적 의미를 그렇게 쉽게 뽑을 수 없다.**

그래서 기존 MIM 연구들은 두 갈래로 갈렸다.

- **항등 사상을 토크나이저로 쓰기** — 픽셀을 그대로 목표로 삼는다. 의미 추상화에 약하고, 고주파 디테일을 모델링하는 데 용량을 낭비한다.
- **미리 학습한 토크나이저 쓰기** — pretrained VAE(DALL-E VAE 등)를 토크나이저로 쓴다. 저수준 의미만 포착되고, 다른 도메인으로 옮기기 어렵다.

두 번째 방식은 **다단계 파이프라인**을 강요한다는 게 더 근본적인 문제다. 목표 모델을 학습하기 전에 의미가 풍부한 토크나이저를 먼저 학습시켜야 한다. 그런데 시각적 의미를 획득하는 것은 어차피 두 단계의 공통 목표다. **그렇다면 따로 할 이유가 있나.**

#### iBOT method

**핵심 제안**
MIM을 **토크나이저로부터의 지식 증류**로 정식화한다. 토크나이저 역할을 하는 twin teacher의 도움을 받아 distillation한다.

- **입력** — 타겟 네트워크(student)는 마스킹된 이미지를, 온라인 토크나이저(teacher)는 원본 이미지를 받는다.
- **목표** — 타겟 네트워크가 마스킹된 각 패치 토큰을, 그 위치에 해당하는 토크나이저 출력으로 복원하게 한다.

**이렇게 해서 풀리는 것**
- 클래스 토큰에 대해 여러 각도의 이미지를 학습시켜 **고수준 시각적 의미**를 포착한다.
- teacher가 momentum update로 MIM과 **공동 최적화**되므로, 전처리 단계에서 별도 학습이 필요 없다.

#### 세부 구조

**전체 흐름**

1. 원본 이미지 `x`에서 augmentation으로 두 개의 view `u`, `v`를 만든다. DINO처럼 global view 2개가 기준이고, 여기에 local view도 추가로 생성해서 CLS loss 쪽에 쓴다.
2. `u`, `v` 각각에 **blockwise masking**을 적용해 masked view `û`, `v̂`를 만든다.
3. **Student network** — masked view `û`, `v̂`를 받아 patch token들의 예측 분포를 출력한다.
4. **Teacher network**(online tokenizer, EMA로 업데이트) — **마스킹 안 된 원본 view** `u`, `v`를 받아 patch token들의 target 분포를 출력한다.

**blockwise masking의 비율**
prediction ratio `r`은 마스킹할 토큰의 비율이다. 논문의 기본 설정은 **`r = 0.3`**이고, 학습 중에는 **확률 0.5로 `r = 0`(마스킹 없음)을 섞고 나머지는 균등 분포에서 뽑는다.** 항상 마스킹하지 않는다는 점이 특징이다.

**MIM Loss (patch 수준)**

```text
L_MIM = -Σᵢ P_teacher(uᵢ) · log P_student(ûᵢ)
        (마스킹된 패치 위치 i 에 대해서만 합산)
```

student가 본 masked view `û`의 각 마스킹 패치 예측이, teacher가 본 원본 view `u`의 같은 위치 패치 출력을 따라가도록 학습한다. `u→û`와 `v→v̂` 양쪽에 대해 동일하게 계산한 뒤 평균낸다. **이 loss는 두 개의 global view에만 적용된다.** local view는 MIM 대상이 아니다.

**CLS Loss (DINO 방식 그대로)**

- student의 CLS 토큰(masked view에서 나온 것)과 teacher의 CLS 토큰(다른 view에서 나온, 마스킹되지 않은 것)을 cross-view로 비교한다.
- DINO의 self-distillation cross-entropy loss를 그대로 쓴다.
- **global view와 local view 전체 조합**에 대해 계산한다. MIM과 달리 local view도 포함된다.

**최종 Loss**

```text
L = L_MIM + L_CLS
```

별도의 가중치 없이 단순 합산한다.

**Projection Head 공유**
CLS 토큰과 patch 토큰의 projection head를 **공유**한다. 파라미터를 따로 두지 않는다.

> DINOv2는 이 부분을 CLS용과 patch용으로 **분리**했다. iBOT loss를 가져오면서 명시한 차이점 중 하나이고, DINOv2 노트의 "MLP head 2개 생성" 부분과 대응된다.
> 흥미로운 건 방향이 규모에 따라 뒤집힌다는 점이다. 작은 규모에서는 공유가 낫고, 크게 키우면 분리가 낫다.

**Architecture**
ViT-S/16, ViT-B/16, ViT-L/16, Swin-T 등 여러 backbone으로 실험한다. **online tokenizer가 별도 pretrain 없이 MIM objective와 동시에 학습된다**는 게 핵심이다. 기존 BEiT류가 미리 학습된 고정 tokenizer(DALL-E VAE 등)를 쓰던 것과 대비된다.

#### 실험에서 확인된 것

- **ImageNet linear probing 82.3%** — ViT-L/16 기준.
- **local semantic pattern이 창발한다.** 논문이 강조하는 부수 관찰인데, 이렇게 학습된 특징이 **강건성**과 dense prediction 태스크(객체 검출, instance segmentation, semantic segmentation)에서의 성능으로 이어진다.

#### 정리

이 논문의 핵심은 **"토크나이저를 미리 학습시켜야 한다"는 전제를 없앤 것**이다.

목표 모델도 토크나이저도 결국 시각적 의미를 얻으려는 것이라면, 둘을 분리할 이유가 없다. EMA teacher가 이미 그 역할을 할 수 있고, 그러면 다단계 파이프라인이 한 단계로 접힌다.

여기서 옮겨갈 만한 발상은 **"전처리 단계의 모델을 학습 루프 안으로 흡수하기"**다. 별도로 준비해야 했던 부품이 사실은 학습 중인 모델 자신으로 대체될 수 있는지 물어보는 것이다.
