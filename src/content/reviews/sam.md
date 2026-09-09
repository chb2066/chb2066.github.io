---
title: SAM
paper: Segment Anything
venue: ICCV 2023
link: https://arxiv.org/abs/2304.02643
claim: 프롬프트로 무엇을 자를지 지정하는 promptable segmentation을 사전학습 과제로 삼으면 zero-shot 전이가 가능한 분할 파운데이션 모델이 된다.
tags: [Segmentation, Foundation Model]
tier: basic
date: 2025-04-28
draft: true
---

#### 목적

이미지를 한 번 인코딩해두고, 다양한 형태의 프롬프트로 원하는 부분만 골라 세그하게 하는 범용 세그멘테이션 파운데이션 모델

#### 주요 용어

foundation model와 downstream task:
- foundation model은 방대한 양의 데이터로 사전 학습(pre-trained)된, 범용적인 능력을 갖춘 거대 AI 모델(ex: SAM)
- downstream task는 위 모델을 통해서 해결하려는 구체적인 task를 의미한다.(ex: 의료 영상 분석, 자율 주행, 사진 편집 등)

#### Abstract

모델을 위해 마스크 10억 개 및 1100만 개의 이미지로 구성된 데이터셋을 구축했다.
프롬프트로 훈련될 수 있게 설계되어 새 이미지 분포, task에 대한 zero-shot transfer가 가능하다.

#### Instruction

웹 수준의 데이터셋에서 pretrain 된 LLM은 제로샷 일반화 성능을 가진다.
핵심 메커니즘은 prompt engineering task를 손으로 만든 텍스트 프롬프트로 표현해주면, 모델이 파인튜닝 없이도 다양한 downstream task에 더욱이 일반화된 결과를 낸다.
이러한 능력은 데이터셋 크기, 모델 크기, 학습 비용이 커질수록 더 좋아진다.
비전 분야에서도 foundation model 시도가 있었으나 대부분 image, text 쌍을 align 하는 방식을 사용한다.
SAM은 segmentation에서도 비슷한 수준의 범용 pretraining task를 정의하는 걸 목표함.

#### Task

모델은 NLP에서 영감을 받았다. 다음 토큰을 예측하는 task가 LLM의 foundation model 역할을 했던 것처럼, segmentation에서도 비슷한 수준의 범용 pretraining task를 정의하는 것을 목표로 한다.
prompt는 전경/배경 점의 집합, 대략적인 box, mask, 자유 형식 텍스트 등 이미지에서 무엇을 segment할지 나타내는 모든 정보를 의미할 수 있다.
promptable segmentation task는 **어떤 프롬프트가 주어져도 유효한 분할 마스크를 반환하는 것**이다. 프롬프트가 모호해서 여러 객체를 지칭할 수 있는 경우에도, 그 중 적어도 하나에 대해서는 합당한 마스크를 출력해야 한다.
NLP 개념을 가져온 이유:
1. 자연스러운 사전학습 알고리즘
1. prompting을 통한 downstream segmentation task로 zero-shot transfer가 가능한 방법

#### **Pre-training**

각 학습 샘플에 대해 순차적인 프롬프트를 시뮬레이션하고, GT와 모델의 mask prediction을 비교한다.
interactive segmentation에서 차용했지만 차이가 있다. 기존 interactive segmentation은 충분한 유저 입력 이후에 유효한 마스크를 점진적으로 예측하는 반면, SAM은 **프롬프트가 애매모호한 시점에도 곧바로 유효 마스크를 예측**하는 것이 목표다.
이렇게 학습해야 data engine이 요구하는 automatic annotation처럼, 모호함을 포함한 사용 사례에서도 모델이 잘 작동한다. 이 task 자체가 난이도가 높아서, 특별히 설계된 모델링과 손실함수가 필요하다.

#### **Zero-shot transfer**

pretraining task를 잘 학습하면, inference 시점에 어떤 프롬프트가 들어와도 적절히 반응할 수 있는 능력이 생긴다. 즉 downstream task는 적절한 prompt engineering만으로 해결 가능해진다.
예를 들어 고양이 bounding box detector가 있다면, 그 탐지 box를 SAM에 프롬프트로 넣어주는 것만으로 고양이 instance segmentation을 풀 수 있다. 이런 식으로 넓은 범위의 실용적인 segmentation task가 prompting으로 변환될 수 있다.
논문에서는 자동 데이터셋 라벨링 외에도 5가지 샘플 task를 통해 이 zero-shot transfer 능력을 보여준다.

#### **Related tasks**

목표는 prompt engineering을 통해 기존 및 새로운 다양한 task에서 두루 쓸 수 있는 모델을 만드는 것이다.
기존 multi-task system과의 차이가 중요하다. multi-task system은 고정된 여러 task를 수행하지만 학습 때 본 task와 테스트 때의 task가 동일하다. 반면 promptable segmentation으로 학습된 SAM은 **학습 시 보지 못한 새로운 task도 inference 시점에 수행**할 수 있다.
이런 특성 덕분에 SAM은 더 큰 시스템의 구성요소로도 쓰일 수 있다. 예를 들어 instance segmentation을 하려면, promptable segmentation 모델을 기존 object detector와 결합하면 된다.

#### SAM model 구성

- Image encoder
  - 고해상도 inputs를 처리하기 위해 최소한으로 적응된 pretrain model된 MAE를 사용한다.
  - 이미지 인코더는 이미지 당 한 번 실행되며 모델에 프롬프트를 주기 전에 적용될 수 있다.
- Prompt encoder
  - 두 종류의 프롬프트를 고려한다: 희소한, 밀집된 성분 포착을 위한 prompts, positional encoding을 사용해서 sparse한 걸 표현하고, 각 프롬프트 타입을 위해 학습된 임베딩과 text encoder인 CLIP에서 나온 자유형 텍스트와 함께.
  - image embedding에 넓은 요쇼를 더하고 conv를 사용하여 Dense prompt는 임베딩 된다.
- Mask decoder
  - mask decode는 이미지 임베딩, prompt embeddings, mask에 대한 출력 token을 효과적으로 매핑한다.
