<!--
개정: 2026-09-10 (원본: src/content/reviews/v-jepa.md)
- ⚠️ 가장 크게 손본 글이다. 원본의 related work 가 논문 원문을 거의 그대로 옮긴 상태였고
  ("Vidusal encoder-decoder", "미세조장할 때", "그것은 그것은", "방ㅂ버들이",
   "정답 벡터 출", "게산된") 오타와 미완성 문장이 많았다. 전부 다시 썼다.
- 원본 마지막이 "EMA로 업데이" 에서 끊겨 있어 완성
- frontmatter 에 paper·venue·link 채움 (arXiv API 로 제목 대조 확인: 2404.08471)
- 원본이 인용 부호로 묶어둔 원문 번역 덩어리는 삭제하고, 그 아래 "→" 로 달아둔
  본인의 요약을 살려 「배경 지식」으로 재구성했다. 요약이 원문보다 정확하다
- 「실험에서 확인된 것」 신설 — ViT-H/16 기준 K400 81.9%, SSv2 72.2%, ImageNet1K 77.9%.
  전부 frozen backbone 기준이라는 점이 이 논문의 요지라 함께 적었다
- method 절의 입력 사양·마스킹 비율·multi-mask 구성은 원본이 정확해서 그대로 유지
- 끝맺음을 평서형으로 통일
-->
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

핵심 키워드:
비디오 SSL, feature prediction, frozen backbone 평가, multi-block masking
주요 전략:
픽셀을 복원하는 대신 target encoder가 만든 잠재 표현을 맞히게 해서, 적응 없이 쓸 수 있는 표현을 얻는다
사용 가능 분야:
동작 인식(Something-Something-v2), 외형 기반 인식(Kinetics-400), 이미지 분류

#### 출발 질문

**시각 세계를 인식하는 능력을 어떻게 학습시킬 수 있는가.** 이 논문은 정적 이미지가 아니라 **비디오로부터 시각적 잠재 표현을 학습**하는 방법을 제안한다.

#### 배경 지식

**갈래 1 — 텍스트 캡션을 쓰는 방법**
CLIP이 대표적이다. 웹에서 긁은 대규모 이미지-텍스트 쌍으로 학습하고, 비디오와 이미지 단위 downstream 태스크에서 강한 성능을 보였다. **백본을 얼린 채 가벼운 태스크 특화 head만 붙여도** 되므로 end-to-end 학습이 필요 없다는 게 장점이다.

비디오 쪽으로도 확장됐다. VideoBERT는 텍스트 캡션을 써서 비디오 인코더가 마스킹된 영역의 표현을 예측하게 학습했고, VideoCLIP은 텍스트 인코더를 쓰면서 비디오 캡션에 대한 잠재 표현으로 대조 학습을 했다. MERLOT, VATT, InternVideo가 이 흐름을 확장한다.

정리하면 **이미지·비디오 캡션으로 잠재 공간이나 마스크 영역을 예측하게 하는 방식이 많았고, 여기에 self-supervised한 오디오·비디오 loss를 더해 높은 성능을 냈다.**

**갈래 2 — 라벨 없이 학습하는 SSL**

SSL은 사람이 라벨링하지 않고도 잠재 표현을 학습한다. 그런데 접근이 다시 둘로 나뉜다.

- **불변성 기반 사전학습** — 손으로 만든 이미지 변형에 대해 불변인 표현을 학습한다. 문제는 **상당한 양의 태스크 특화 귀납 편향을 요구**한다는 것이고, 그만큼 적용 범위가 제한된다.
- **재구성 기반** — DAE는 손상된 입력을 복원하는 것으로 표현을 학습할 수 있음을 보였다. MAE는 encoder-decoder가 마스킹된 이미지 패치의 **픽셀**을 예측하게 한다.

MAE는 파인튜닝하면 성능이 좋다. 하지만 **MAE가 만든 잠재 표현은 적응 절차를 거쳐야 하고, 얼린 상태로는 불변성 기반 표현에 미치지 못한다.** 그래서 다른 태스크에서는 **날 픽셀보다 잠재 공간에서 예측하는 쪽**으로 방향이 잡혔다.

DINOv2는 불변성 기반 loss와 masked image modeling loss를 결합해 SSL에서 경쟁력을 보였고, CLIP처럼 텍스트를 쓰지 않아도 된다는 것을 보여줬다. 다만 **정적 이미지 학습에 머물러 있어서 동적인 정보를 수집하는 것은 기대할 수 없다.**

> 요약하면 masking 기반 방법들이 불변 잠재 공간을 다룰 때 좋은 성능을 보인 사례가 많다.
> 하지만 대부분 정적인 데 머물고, 동적인 경우의 정보 수집 능력은 상대적으로 떨어진다.
> **V-JEPA는 그 지점을 다룬다.**

**갈래 3 — 비디오에서의 SSL**

비디오 쪽에서도 여러 시도가 있었다.

- 시간에 걸쳐 불변인 잠재 공간을 학습하기
- 다음 프레임의 잠재 표현을 계층 구조로 학습하기
- segmentation을 써서 강한 잠재 표현으로 이어가기
- MAE 접근을 확장해 **마스킹된 시공간 복셀**을 예측하는 encoder-decoder를 학습하기

사전학습된 CLIP encoder의 **얼린 잠재 공간에서 계산한** 마스크 모델링 loss로 모델을 학습한 사례도 있다. **V-JEPA는 얼린 외부 인코더를 쓰지 않고, 학습 중에 온라인으로 잠재 공간을 예측한다는 점이 다르다.**

이미지와 영상을 함께 학습시킨 연구도 있고, MAE에 계층적 Transformer 구조를 쓴 이점을 설명한 연구, 프레임 레벨 인코더로 MAE를 확장하거나 cross-attention 기반 video decoder로 확장한 연구도 있다. MAE 기반 접근은 사전학습 동안 편향을 최소한으로 만들고 파인튜닝 시 downstream 성능이 좋다. **그러나 V-JEPA는 더 나은 잠재 표현으로 이어진다는 것을 보인다.**

> 결국 SSL은 두 갈래다. 하나는 **이미지 변형에 무관하게 특징을 뽑는 불변성 학습**이고,
> 다른 하나는 **가려진 부분을 맞히는 재구성 기반 학습**이다.

#### V-JEPA method

**JEPA의 기본 아이디어**
한 곳을 보고 다른 곳을 예측하는 방식이다. 네트워크 세 개를 쓴다.

- **context encoder** — context에 대한 잠재 표현을 계산한다.
- **target encoder** — 타겟의 잠재 표현을 계산한다.
- **predictor** — 둘 사이의 상대적 위치 정보를 받아, context representation으로부터 target representation을 예측한다.

세 네트워크의 목적은 **predictor의 출력과 target encoder의 출력 사이의 손실을 최소화**하는 것이다.

V-JEPA는 여기에 마스킹을 쓴다. context encoder와 predictor가 마스킹된 비디오를 처리해서 마스킹 영역의 내용을 예측하고, 그 예측을 **마스킹되지 않은 비디오를 본 target encoder의 출력과 비교**한다. loss는 L1이다.

**세 네트워크의 역할**

| | context encoder | target encoder | predictor |
|---|---|---|---|
| 입력 | 마스킹되고 남은 패치 | 원본 비디오 전체 | context token + mask token |
| 보는 것 | 비디오의 일부 | 비디오 전체 | — |
| 갱신 | predictor의 loss로 학습 | **EMA로 갱신** | loss로 학습 |
| 출력 | 보이는 부분의 context representation | 정답 벡터 | 맥락 기반 예측 |

![그림 1](/img/v-jepa/01.png)

#### 입력 처리

- 비디오에서 랜덤 시작점으로 **연속 64프레임 클립**을 뽑고, temporal stride 4로 **16프레임을 균등 샘플링**한다.
- 16프레임 클립에 **3D conv(2×16×16 필터 d개)**를 적용해 `8×14×14×d` 텐서를 만든다.
- **3D positional embedding**을 더하고 flatten해서 `1568×d` 토큰 시퀀스로 만든다.

**Multi-block masking**

- **Short-range** — 8개 블록의 합집합이 프레임 면적의 약 15%를 덮는다.
- **Long-range** — 2개 블록의 합집합이 약 70%를 덮는다.
- 종횡비는 0.75~1.5에서 랜덤하게 뽑는다.
- **공간 마스크를 시간축 전체에 반복 적용**한다. temporal mask ratio가 사실상 1.0이다.
- 최종 마스킹 비율은 **약 90%**다.

시간축 전체에 같은 공간 마스크를 적용한다는 게 중요하다. 그렇게 하지 않으면 다른 프레임에서 그대로 답을 볼 수 있어서 과제가 너무 쉬워진다.

#### 학습

**Patch 수준 loss**

1. 비디오 클립을 마스킹해 context encoder에 넣고 context representation을 만든다. 마스킹된 클립을 넣는다는 것은 **남아 있는 패치의 잠재 표현이 타겟 영역을 예측하도록** 만드는 것이다.
2. predictor는 context encoder가 만든 토큰을 받아 비디오 클립에서 사라진 영역을 예측한다. **학습 가능한 mask token**을 함께 넣는데, mask token은 **공유된 학습 가능 벡터와 3D sin-cos positional embedding의 합**으로 파라미터화된다.
3. predictor의 예측과 target encoder의 출력을 L1 loss로 비교한다.

정리하면 **predictor가 가려진 부분을 예측하고, target encoder가 전체 영상을 정답지로 만들며, L1 loss로 둘의 차이를 줄인다. EMA가 정답지 기준을 안정화한다.**

**Multi-Mask Prediction**
같은 비디오에 대해 **0.15짜리 마스크를 8번, 0.7짜리 마스크를 2번** 각각 독립적으로 예측한다. 각 결과를 정답과 비교하고 오차들의 평균으로 loss를 구성한다.

**붕괴 방지**
**stop-gradient**를 쓴다. target encoder로는 gradient가 흐르지 않고 EMA로만 갱신된다. 이 비대칭이 표현이 한 점으로 무너지는 것을 막는다.

#### 실험에서 확인된 것

가장 큰 모델인 **ViT-H/16을 비디오만으로 학습**한 결과다. 전부 **파라미터를 전혀 조정하지 않은 frozen backbone** 기준이다.

| 태스크 | 성능 |
|---|---|
| Kinetics-400 (외형 기반) | **81.9%** |
| Something-Something-v2 (동작 기반) | **72.2%** |
| ImageNet-1K | **77.9%** |

이 표의 요지는 숫자 자체가 아니라 **같은 얼린 백본 하나로 성격이 다른 두 태스크를 모두 처리했다**는 데 있다. Something-Something-v2는 시간적 동작을 이해해야 풀리고, Kinetics-400은 외형만으로도 상당 부분 풀린다. 보통은 한쪽에 맞추면 다른 쪽이 약해진다.

그리고 **frozen 평가(attentive probing) 프로토콜에서 픽셀 예측 방식들을 앞선다.** 앞서 배경에서 짚은 "MAE의 표현은 얼린 상태로는 약하다"는 문제를 실제로 넘어선 것이다.

#### 정리

이 논문의 핵심은 **"무엇을 예측 목표로 삼을 것인가"**다.

픽셀을 목표로 삼으면 저수준 디테일을 맞히는 데 용량이 쓰이고, 얻어진 표현은 그대로 쓰기 어렵다. 잠재 표현을 목표로 삼으면 **무엇을 버릴지를 target encoder가 알아서 정한다.** 그리고 그 target encoder는 EMA로 천천히 따라오는 자기 자신이라, 외부 인코더에 의존하지 않는다.

I-JEPA가 이미지에서 한 일을 비디오로 옮긴 것이고, 옮기면서 추가된 설계는 **시간축 전체에 같은 공간 마스크를 적용하는 것**이다.
