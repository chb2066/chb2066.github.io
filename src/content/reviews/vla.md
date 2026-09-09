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

#### Input

- image, txt
  - 이미지+ 언어 지시를 통해 이미지에서 해야할 행동을 입력
    - ex): ‘주방’ image, ”pick up the cup” txt 입력하여 주방 내에 컵을 집는 행동을 시킴)

#### Input 구조

- Vision model(dinov2, siglp)로 이미지 임베딩, llama에 있는 토크나이저로 언어 지시 임베딩 후 llama 모델에 넣음
- 이미지 패치 토큰 + 언어 지시 토큰을 concat 후 LLM backbone에 입력하는 구조

#### SigLIP 쓰는 이유

- VLM은 image, txt를 사용, 이때 로봇의 action은 image 내에서 txt의 언어 지시를 받아 행동해야한다. 따라서 image를 임베딩 받을 때 해당 임베딩 벡터는 텍스트의 영향을 받을 경우 도 좋은 성능을 낸다.
- SigLIP은 image-text contrastive 학습으로 patch feature 공간이 텍스트와 의미적으로 정렬(align)되도록 학습됨. 따라서 VLM에 입력되는 이미지 patch token도 텍스트와 정렬된 임베딩 벡터.
- 따라서 SigLIP를 사용하여, 이미지가 가지는 텍스트적 성질이 반영된 토큰을 LLM에 입력하기 위해 해당 모델을 사용한다.
SigLIP 특징:
CLIP은 배치 내 모든 쌍에 대해 softmax 기반 contrastive loss를 사용해서 큰 배치가 필요한 반면, SigLIP은 각 쌍을 독립적인 binary classification 문제로 보는 sigmoid loss를 사용해서 배치 전체 정규화가 필요 없고
작은 배치에서도 안정적으로 학습 가능함.

#### Action 만드는 구조

실제 action 값:  [0.023, -0.11, 0.87, 0.002, -0.45, 0.33, 1.0]
(↑ 7-DOF (x,y,z, roll,pitch,yaw, gripper)
문제점
LLM은 토큰(정수)만 생성할 수 있으나 action값은 연속값(실수) 이다.
해결: 구간을 256칸으로 나누기
이동 범위가 q1~q99
q1                              0                           q99
|----|----|----|----|----|----|----|----|---|
0     1      2     3    ... 128 ...    253 254 255
실수값이 어느 칸에 속하는 지를 통해 실수값을 다룸.  학습 데이터 action 분포의 1~99 percentile 구간 [q1, q99]을 256개 bin으로 나눈다. 분포 양 끝의 outlier 때문에 균등 binning을 쓰면 대부분의 실제 action 값이 몰려있는 구간의 bin 해상도가 낭비되는 문제가 있어서, percentile 기반 quantile binning을 사용.
ex)

---

x = 0.023
학습 데이터에서 이 액션 차원의 [q1, q99] = [-0.8, 0.9]였다고 하면 이 범위를 256개 bin으로 균등 분할
→ 0.023이 속하는 bin index 계산
→ bin index = 152

---

해당 방식으로 action 값을 다룸.

#### Output 구조

LLM이 autoregressive 하게 토큰 생성
→ token_1 (x축 이동)
→ token_2 (y축 이동)
→ token_3 (z축 이동)
→ token_4 (roll)
→ token_5 (pitch)
→ token_6 (yaw)
→ token_7 (gripper open/close)
각 토큰 → bin index → 연속값 역변환 → 로봇 관절에 전달(bin index는 tokenizer에서 덜 쓰이는 토큰 256개를 활용)

#### Loss 구성(CE 만 사용)

Loss= CE(action 토큰 7개 평균)
= (CE_x + CE_y + CE_z + CE_roll + CE_pitch + CE_yaw + CE_gripper) / 7
