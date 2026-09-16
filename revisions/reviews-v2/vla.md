---
title: OpenVLA
paper: "OpenVLA: An Open-Source Vision-Language-Action Model"
venue: CoRL 2024
authors: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al.
link: https://arxiv.org/abs/2406.09246
claim: 이미지와 언어 지시를 VLM 백본에 넣고, 연속적인 로봇 action을 이산 토큰으로 양자화해 autoregressive하게 생성한다.
tags: [Vision-Language, Robotics]
tier: main
date: 2025-08-06
draft: false
---

> 📄 [**OpenVLA: An Open-Source Vision-Language-Action Model**](https://arxiv.org/abs/2406.09246) · CoRL 2024 · Kim, Pertsch, Karamcheti et al.

## Abstract

**문제**
로봇 정책은 학습한 물체·장면·태스크 밖으로 잘 일반화되지 않음. 인터넷 규모로 사전학습된 VLM은 그 일반화를 갖고 있지만, 로봇 제어로 가져오는 경로가 닫혀 있었음

**해결책**
Open X-Embodiment 97만 궤적으로 VLM을 fine-tuning해 7B VLA 정책을 만들고, 가중치·학습 코드·fine-tuning 레시피를 전부 공개
연속 action을 이산 토큰으로 바꿔 언어모델 백본을 구조 변경 없이 그대로 사용

---

## 1. Introduction

![Figure 1](/img/vla/f1.png)

**VLA(Vision-Language-Action)란**: 이미지 + 언어 지시 → 로봇 행동. 매번 처음부터 학습시키는 대신, 인터넷 규모 vision-language 데이터로 사전학습된 모델을 로봇 시연으로 fine-tuning해 일반화되는 정책을 얻는 흐름

**기존 상황**
- CLIP, SigLIP, Llama 2 같은 모델이 보여준 일반화를 로봇 제어로 옮기려는 시도가 이어짐
- 가장 가까운 선행 연구는 **RT-2-X** — Open X-Embodiment로 학습한 55B VLA, 당시 최고 성능

**RT-2-X의 두 가지 문제**
- 모델이 닫혀 있음 → 가중치도 학습 코드도 비공개라 이어서 연구할 수 없음
- 새 태스크로 효율적으로 fine-tuning하는 방법이 탐색되지 않음 → 실제로 쓰려면 이게 핵심인데 비어 있었음

**본 논문의 기여**
1. 97만 궤적으로 학습한 7B VLA를 가중치·코드·레시피까지 전부 공개
1. LoRA fine-tuning과 양자화 서빙으로 소비자용 GPU에서 쓸 수 있게 함
1. RT-2-X(55B) 대비 절대 성공률 +16.5%p — 파라미터는 7배 적음

---

## 2. Related Work

**시각-언어 모델의 로봇 전이**
- 초기: 사전학습 시각 인코더만 가져다 쓰고 정책은 처음부터 학습
- 이후: VLM 전체를 백본으로 쓰고 action을 출력하게 fine-tuning → RT-2 계열

**행동을 어떻게 출력하는가로 갈림**
- **연속값 회귀** — 별도 action head를 붙여 실수를 직접 예측. 언어모델 구조를 바꿔야 함
- **이산 토큰 생성** — action을 토큰으로 바꿔 언어모델이 그대로 생성. RT-2와 본 논문이 여기 속함

**차별점**
- 새 알고리즘이 아니라 **구성의 선택**에 초점 — 인코더 조합, 이산화 방식, 데이터 큐레이션
- 공개 범위 — 선행 연구가 닫아둔 가중치·코드·fine-tuning 레시피를 전부 염

---

## 3. The OpenVLA Model

이미지와 언어 지시를 받아 로봇 행동을 낸다. 주방 이미지와 `pick up the cup`을 넣으면 컵을 집는 행동이 나온다.

핵심은 문제를 **vision-language 태스크로 재정의**한 것. 관측 이미지와 자연어 지시를 받아 "예측된 로봇 행동의 문자열"을 출력하는 문제로 보면, VLM의 언어모델 백본을 구조 변경 없이 그대로 쓸 수 있다.

### 3.1 Preliminaries: Vision-Language Models

![Figure 2](/img/vla/f2.png)

**입력 요소**
- **이미지** — vision encoder를 통과해 패치 토큰이 됨
- **언어 지시** — Llama tokenizer로 토큰화
- 두 토큰열을 concat해서 LLM backbone에 넣음

**백본: Prismatic-7B VLM**
- **visual encoder** — SigLIP과 DINOv2 두 개. 같은 이미지 패치를 양쪽에 통과시킨 뒤 feature를 **채널 방향으로 concat**
- **projector** — 융합된 시각 feature를 언어모델 임베딩 차원으로 사상하는 2층 MLP
- **LLM backbone** — Llama 2 7B

**SigLIP을 쓰는 이유**
- 로봇의 action은 이미지 안에서 텍스트 지시를 받아 수행됨 → 이미지 임베딩이 **텍스트의 영향을 받은 상태**인 편이 유리
- SigLIP은 image-text contrastive 학습으로 patch feature 공간이 텍스트와 의미적으로 정렬되도록 학습됨 → LLM에 들어가는 패치 토큰이 이미 텍스트와 정렬된 벡터

**SigLIP과 CLIP의 차이**
- CLIP — 배치 내 모든 쌍에 softmax 기반 contrastive loss. 큰 배치가 필요
- SigLIP — 각 쌍을 독립적인 binary classification으로 보는 sigmoid loss. 배치 전체 정규화가 없어 작은 배치에서도 안정적

**DINOv2를 함께 쓰는 이유**
- DINOv2의 저수준 공간 정보 + SigLIP의 고수준 의미 → 공간 추론 개선
- 로봇 조작은 "무엇을"뿐 아니라 "**어디를**"이 중요 → 이 조합이 맞아떨어짐

### 3.2 OpenVLA Training Procedure

**행동 이산화** — 연속 행동을 LLM이 다룰 수 있는 정수 토큰으로 바꾼다.

- action 값 예시: `[0.023, -0.11, 0.87, 0.002, -0.45, 0.33, 1.0]` — 7-DOF (x, y, z, roll, pitch, yaw, gripper)
- LLM은 토큰만 생성할 수 있는데 action은 연속값 → **각 차원을 독립적으로 256개 bin에 이산화**
- bin 폭은 학습 데이터 action 분포의 **1st ~ 99th quantile 구간을 균등 분할**해서 정함

```
       q1                          q99
       |----|----|----| ... |----|----|
        0    1    2    ...  254  255
```

- **quantile을 쓰는 이유**: min-max로 잡으면 분포 양 끝의 outlier가 구간을 크게 벌려서, 실제 action이 몰려 있는 구간의 해상도가 낭비됨. RT-2가 min-max를 쓰던 부분을 바꾼 것

**토큰 자리 확보**
- Llama tokenizer는 새로 도입되는 토큰용 special token을 100개만 예약 → 256개가 필요한데 부족
- 어휘 사전에서 **가장 덜 쓰이는 토큰 256개(마지막 256개)를 action 토큰으로 덮어씀**

**출력**
```
token_1 → x축 이동      token_5 → pitch
token_2 → y축 이동      token_6 → yaw
token_3 → z축 이동      token_7 → gripper open/close
token_4 → roll
```
- 각 토큰을 bin index로 읽고 연속값으로 역변환해 로봇 관절에 전달

**Loss**: 표준 next-token prediction을 쓰되 **cross-entropy를 action 토큰에 대해서만** 계산
```
Loss = (CE_x + CE_y + CE_z + CE_roll + CE_pitch + CE_yaw + CE_gripper) / 7
```

### 3.3 Training Data

**Open X-Embodiment**
- 70개 이상의 개별 로봇 데이터셋, 200만 개 이상의 궤적이 하나의 형식으로 모여 있음
- 큐레이션을 거쳐 **97만 궤적**을 학습에 사용

**큐레이션의 목적 두 가지**
1. 모든 학습 데이터에서 **입력·출력 공간을 일관되게** 만들기
1. embodiment, 장면, 태스크의 **다양성 확보**

### 3.4 OpenVLA Design Decisions

설계 선택을 하나씩 비교해 고른 절. 값 자체보다 **무엇을 비교했는지**가 요점이다.

- **이미지 해상도** — 올려도 성능이 오르지 않았고 학습 시간만 늘었음. VLM 벤치마크에서는 해상도를 올리면 좋아지는 경우가 많은데 여기서는 그렇지 않았다는 게 특징
- **vision encoder fine-tuning 여부** — 얼리지 않고 함께 학습시키는 편이 나았음. 로봇 제어에 필요한 공간 정보가 사전학습만으로는 부족하다는 뜻
- **학습 epoch** — 언어모델 학습의 통상 범위보다 훨씬 오래. action token 정확도가 충분히 올라갈 때까지 돌림
- **learning rate** — 고정값이 warmup보다 나았음

---

## 4. Experiments

**여러 로봇 플랫폼에서의 직접 평가**
- **RT-2-X(55B) 대비 절대 성공률 +16.5%p** — 29개 태스크, 여러 로봇 embodiment에서. 파라미터는 7배 적음
- **Diffusion Policy 대비 +20.4%p** — 여러 물체가 등장하고 강한 language grounding이 필요한 다중 태스크 환경에서 특히 강했음

**새 로봇 셋업으로의 적응**

![Figure 5](/img/vla/f5.png)

- 적은 시연만으로 새 셋업에 맞춰짐 → 실제로 쓰려면 이게 핵심인데 RT-2-X에는 비어 있던 부분

**효율화**
- **LoRA fine-tuning** — 소비자용 GPU에서 새 로봇 셋업으로 적응 가능
- **양자화 서빙** — 메모리를 줄여도 downstream 성공률이 떨어지지 않음

![Table 2](/img/vla/t2.png)

---

## 정리

이 논문의 값어치는 새로운 알고리즘이 아니라 **구성 변경**에 있다.

- RT-2의 행동 토큰화를 가져오되 binning을 min-max에서 quantile로 변경.
- 인코더를 SigLIP 단독이 아니라 DINOv2와 융합해서 공간 정보를 보강.
- 결과 **7배 작은 모델로 55B급 모델을 이김.**

---

*그림은 모두 원 논문에서 가져왔다. Kim et al., [OpenVLA](https://arxiv.org/abs/2406.09246), CoRL 2024.*
