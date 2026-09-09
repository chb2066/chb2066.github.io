---
title: U-Net
paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
venue: MICCAI 2015
link: https://arxiv.org/abs/1505.04597
claim: 대칭 encoder-decoder에 skip connection을 넣어, 적은 의료 영상 데이터로도 경계까지 복원하는 segmentation이 가능하다.
tags: [Segmentation, Medical]
tier: basic
date: 2025-03-17
draft: true
---

## Introduction

- 의료 영상에서는 데이터가 부족한 경우가 많아 이를 위한 segmentation을 위한 FCN 기반 모델 제안.
- Encoder-Decoder 구조로 전역 정보와 세부 정보 활용
- Date augmentation을 적극 사용해 적은 데이터로 효과적 학습 가능.
- Skip connection 사용

## Network Architecture

#### 구조 및 특징:

1. 대칭 U자형 구조
1. FC Layer없이 conv 연산만을 사용.(공간 정보 소실 막음)
1. 다운샘플링 경로에서 추출된 고해상도 특징을 업샘플링 경로와 결합하여 세밀한 정보를 복원(작은 이미지 정보)
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

Encoder에서 CNN과정을 통해 이미지에 대한 전역적인 문맥을 학습하고 Decoder 과정에서 해상도를 복원하고 경계를 복원할 수 있다.
**Encoder(축소) Contraction Path(Down 과정):**
1. Conv 3*3+ReLU 2번
1. max pool 2*2
**Decoder(복원) Expanding Path(UP 과정):**
1. feature map을 2*2 up-conv로 upsampling한다.
1. skip connection, down 과정에서 나온 feature map을 upsampling 된 feature map에 concat한다.
  1. 이때 두 이미지 간 크기가 다르기 때문에 upsampling된 이미지에 패딩을 더하여 진행한다.
1. Conv 3*3+ReLU 2번(첫 conv에서는 채널 수를 맞춰 세부정보를 복원한다. )
1. 1×1 Convolution을 사용하여 최종 클래스 개수만큼 채널 수 조정
**Output:**
softmax를 적용해 픽셀 단위의 확률 값을 출력해 segmentation을 진행한다.

---

**Unet의 skip connection:**
인코더의 고해상도 특징을 디코더에서 업샘플된 출력과 결합하고 이후 conv layer가 이를 조합하는 것
**Encoder, Decoder 과정의 문제점:**
- Downsampling을 반복함에 따라 feature map이 축소되어 세밀한 정보나 픽셀의 위치 정보가 손실된다.
- conv 과정 중 경계 정보 손실 발생한다.
→업샘플링 시 경계 정보, 위치 정보, 세밀한 정보 등이 부족하여 흐릿한 결과가 생성된다.
**Skip connection을 통한 해결:**
skip connection을 하게 되면 upsampling 된 feature map에 인코더의 feature map을 더하여 위에서 발생하는 다양한 정보의 손실 등을 완화할 수 있다.

## Training

**학습 설정:**
SGD 사용 , GPU memory한계 때문에 Overlap-Tile Strategy을 통해 large input tiles 사용(더 많은 문맥 정보 습득 가능)하고 batch_size를 1, Momentum 0.99 사용(배치크기가 작아서 이전 학습 샘플 영향 크게 받음.)
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

**손실 함수:**
- 픽셀 단위 Softmax+ Cross Entropy loss 사용
- Weight Map을 추가하여 경계를 강조 (Boundary Loss 적용)
  > *[그림 자리 — Notion 원본에서 옮겨야 함]*

- 같은 클래스 내 인접한 객체들이 붙어 있는 경우, 경계를 학습하도록 추가 가중치 부여
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

**가중치 초기화:**
네트워크의 Feature Map이 단위 분산을 유지하도록 초기 가중치 조정.
He Initialization 사용:
표준편차 루트2/N을 사용하여 초기 가중치 샘플** **
