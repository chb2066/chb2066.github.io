<!--
개정: 2026-09-10 (원본: src/content/reviews/vla.md)
- 논문을 OpenVLA(arXiv 2406.09246)로 확정하고 frontmatter에 서지 정보 반영
- 원본에 없던 「배경 지식」 추가 — RT-2 계열이 무엇을 했고 무엇이 닫혀 있었는지
- 「세부 아키텍처」에 Prismatic-7B 기반, DINOv2+SigLIP 융합 방식, 학습 설정(64×A100 14일,
  224px, 27 epoch, lr 2e-5) 보강. 전부 논문 3.1~3.4절에서 확인
- 「실험에서 확인된 것」 신설 — RT-2-X(55B) 대비 +16.5%p, Diffusion Policy 대비 +20.4%p,
  LoRA·양자화 결과. 논문 4절
- 끝맺음을 평서형으로 통일
- 원본의 SigLIP 선택 이유에 대한 판단은 그대로 유지. 논문 근거(DINOv2가 공간 추론을
  보강한다는 서술)를 덧붙임
-->
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

핵심 키워드:
open-source VLA, discrete action tokenization, DINOv2+SigLIP 융합 인코더, Open X-Embodiment
주요 전략:
연속 행동을 quantile binning으로 256개 bin에 이산화하고, Llama tokenizer의 덜 쓰이는 토큰 256개를 덮어써서 행동을 "말하게" 만든다
사용 가능 분야:
다중 로봇 조작, 새 로봇 셋업으로의 fine-tuning, 소비자용 GPU에서의 적응

#### 배경 지식

**VLA가 풀려는 문제**
로봇에게 새 기술을 가르칠 때 매번 처음부터 학습시키는 대신, 인터넷 규모의 vision-language 데이터로 사전학습된 모델을 로봇 시연으로 fine-tuning해서 일반화되는 정책을 얻자는 것이다. CLIP, SigLIP, Llama 2 같은 모델이 보여준 일반화 능력을 로봇 제어로 가져오려는 흐름이다.

**직전 연구와 무엇이 달랐나**
가장 가까운 선행 연구는 RT-2-X다. Open X-Embodiment 데이터로 학습한 55B 파라미터 VLA 정책이고, 당시 최고 성능이었다. 그런데 두 가지 문제가 있었다.

1. **모델이 닫혀 있다.** 가중치도 학습 코드도 공개되지 않아서 아무도 이어서 연구할 수 없다.
2. **새 태스크로 효율적으로 fine-tuning하는 방법이 탐색되지 않았다.** 실제로 쓰려면 이게 핵심인데 비어 있었다.

OpenVLA는 이 둘을 정면으로 다룬다. 7B 파라미터로 55B를 이기면서 전부 공개한다.

#### OpenVLA method

**입력과 출력**
이미지와 언어 지시를 받아 로봇이 해야 할 행동을 낸다. 예를 들어 주방 이미지와 `pick up the cup`이라는 지시를 넣으면 컵을 집는 행동이 나온다.

문제를 **vision-language 태스크로 재정의한 것**이 핵심이다. 관측 이미지와 자연어 지시를 입력으로 받아 "예측된 로봇 행동의 문자열"을 출력하는 문제로 본다. 그러면 VLM의 언어 모델 백본을 구조 변경 없이 그대로 쓸 수 있다.

**입력 구조**
- vision encoder(DINOv2, SigLIP)로 이미지를 임베딩하고, Llama tokenizer로 언어 지시를 임베딩한다.
- 이미지 패치 토큰과 언어 지시 토큰을 concat해서 LLM backbone에 넣는다.

#### SigLIP을 쓰는 이유

VLM은 이미지와 텍스트를 함께 다루는데, 로봇의 action은 이미지 안에서 텍스트 지시를 받아 수행돼야 한다. 그래서 이미지를 임베딩할 때 그 벡터가 **텍스트의 영향을 받은 상태**인 편이 유리하다.

SigLIP은 image-text contrastive 학습으로 patch feature 공간이 텍스트와 의미적으로 정렬되도록 학습된다. 따라서 VLM에 들어가는 이미지 patch token도 텍스트와 정렬된 벡터다. 이미지가 가진 텍스트적 성질이 반영된 토큰을 LLM에 넣기 위해 이 모델을 쓴다.

**SigLIP과 CLIP의 차이**
CLIP은 배치 내 모든 쌍에 대해 softmax 기반 contrastive loss를 써서 큰 배치가 필요하다. SigLIP은 각 쌍을 독립적인 binary classification 문제로 보는 sigmoid loss를 써서 배치 전체 정규화가 필요 없고, 작은 배치에서도 안정적으로 학습된다.

**그런데 SigLIP만 쓰지 않는다**
논문은 CLIP이나 SigLIP 단독 인코더 대신 **DINOv2를 함께 융합**한다. DINOv2의 저수준 공간 정보가 SigLIP의 고수준 의미와 합쳐지면 공간 추론이 개선된다는 것이 근거다. 로봇 조작은 "무엇을"뿐 아니라 "어디를"이 중요하므로 이 조합이 맞아떨어진다.

#### 세부 아키텍처

**백본: Prismatic-7B VLM**
- **visual encoder** — SigLIP과 DINOv2 두 개. 입력 이미지 패치를 양쪽에 통과시킨 뒤 **feature를 채널 방향으로 concat**한다.
- **projector** — 융합된 시각 feature를 언어 모델의 임베딩 차원으로 사상하는 2층 MLP.
- **LLM backbone** — Llama 2 7B.

**행동 이산화**
연속 행동을 LLM이 다룰 수 있는 정수 토큰으로 바꾼다.

- 실제 action 값 예시: `[0.023, -0.11, 0.87, 0.002, -0.45, 0.33, 1.0]` — 7-DOF (x, y, z, roll, pitch, yaw, gripper).
- LLM은 토큰(정수)만 생성할 수 있는데 action은 연속값이다. 그래서 **각 차원을 독립적으로 256개 bin에 이산화**한다.
- bin 폭은 학습 데이터 action 분포의 **1st ~ 99th quantile 구간을 균등 분할**해서 정한다.

```
       q1                          q99
       |----|----|----| ... |----|----|
        0    1    2    ...  254  255
```

quantile을 쓰는 이유가 중요하다. min-max로 잡으면 분포 양 끝의 outlier가 구간을 크게 벌려서, 실제 action이 몰려 있는 구간의 해상도가 낭비된다. RT-2가 min-max를 썼던 부분을 quantile로 바꾼 것이다.

예를 들어 `x = 0.023`이고 이 차원의 `[q1, q99]`가 `[-0.8, 0.9]`였다면, 이 범위를 256개 bin으로 균등 분할해서 bin index를 계산한다.

**토큰 자리를 어떻게 확보하는가**
Llama tokenizer는 fine-tuning 중 새로 도입되는 토큰용으로 **special token을 100개밖에 예약해두지 않는다.** 256개가 필요한데 부족하다. 그래서 **어휘 사전에서 가장 덜 쓰이는 토큰 256개(마지막 256개)를 action 토큰으로 덮어쓴다.**

**출력 구조**
LLM이 autoregressive하게 토큰을 생성한다.

```
token_1 → x축 이동      token_5 → pitch
token_2 → y축 이동      token_6 → yaw
token_3 → z축 이동      token_7 → gripper open/close
token_4 → roll
```

각 토큰을 bin index로 읽고 연속값으로 역변환해 로봇 관절에 전달한다.

**Loss**
표준 next-token prediction 목적함수를 쓰되, **cross-entropy를 action 토큰에 대해서만** 계산한다.

```
Loss = (CE_x + CE_y + CE_z + CE_roll + CE_pitch + CE_yaw + CE_gripper) / 7
```

#### 학습 데이터와 설정

**Open X-Embodiment**
70개 이상의 개별 로봇 데이터셋, 200만 개 이상의 궤적이 하나의 형식으로 모여 있는 데이터다. 여기서 큐레이션을 거쳐 **970k 궤적**을 학습에 쓴다. 큐레이션의 목적은 두 가지다.

1. 모든 학습 데이터에서 **입력·출력 공간을 일관되게** 만들기
2. embodiment, 장면, 태스크의 다양성 확보

**학습 설정** (논문 3.4절의 설계 결정)

| 항목 | 값 | 근거 |
|---|---|---|
| 이미지 해상도 | 224×224 | 384×384와 성능 차이가 없는데 학습이 3배 오래 걸린다 |
| epoch | 27 | action token 정확도가 95%를 넘을 때까지. LLM 학습의 통상 1~2 epoch와 크게 다르다 |
| learning rate | 2e-5 고정 | 여러 자릿수를 훑어서 결정. warmup은 이득이 없었다 |
| 하드웨어 | 64× A100, 14일 | |

VLM 벤치마크에서는 해상도를 올리면 성능이 오르는 경우가 많은데, **여기서는 그렇지 않았다**는 게 특기할 만하다.

#### 실험에서 확인된 것

- **RT-2-X(55B) 대비 절대 성공률 +16.5%p.** 29개 태스크, 여러 로봇 embodiment에서. **파라미터는 7배 적다.**
- **Diffusion Policy 대비 +20.4%p.** 여러 물체가 등장하고 강한 language grounding이 필요한 다중 태스크 환경에서 특히 강했다.
- **LoRA로 소비자용 GPU에서 fine-tuning이 가능하고, 양자화해서 서빙해도 downstream 성공률이 떨어지지 않는다.** 이게 별도 기여로 제시된다.

#### 정리

이 논문의 값어치는 새로운 알고리즘이 아니라 **선택의 조합**에 있다.

- RT-2의 행동 토큰화를 가져오되 binning을 min-max에서 quantile로 바꿨다.
- 인코더를 SigLIP 단독이 아니라 DINOv2와 융합해서 공간 정보를 보강했다.
- 그 결과 7배 작은 모델로 55B를 이겼고, 전부 공개했다.

행동을 토큰으로 바꾸는 순간 **LLM 생태계 전체를 상속받는다**는 게 이 계열의 핵심이다. 디코딩 전략, 캐싱, 양자화, LoRA가 전부 그대로 따라온다.
