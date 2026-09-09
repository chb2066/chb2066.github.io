# 리뷰 개정 진행 상황

형식은 [FORMAT.md](FORMAT.md). 개정본은 `revisions/reviews/<slug>.md` 로 만든다.
**원본 `src/content/reviews/` 는 건드리지 않는다.**

작업할 때마다 **위에서부터 `대기` 인 것 하나**를 골라 처리하고 이 표를 갱신한다.
한 번에 한 편만 한다. 서두르지 않는다.

## 우선순위 1 — 공개 중 (11편)  ✅ 전부 완료

지금 사이트에 떠 있는 글들이라 먼저 손본다.

| # | slug | 논문 | arXiv | 상태 |
|---|---|---|---|---|
| 1 | `i-jepa` | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture | 2301.08243 | **기준 형식** — 개정 불필요 |
| 2 | `vla` | OpenVLA: An Open-Source Vision-Language-Action Model | 2406.09246 | **완료** |
| 3 | `personaplex` | PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models | 2602.06053 | **완료** |
| 4 | `dinov2` | DINOv2: Learning Robust Visual Features without Supervision | 2304.07193 | **완료** |
| 5 | `dit` | Scalable Diffusion Models with Transformers | 2212.09748 | **완료** |
| 6 | `ibot` | iBOT: Image BERT Pre-Training with Online Tokenizer | 2111.07832 | **완료** |
| 7 | `clip` | Learning Transferable Visual Models From Natural Language Supervision | 2103.00020 | **완료** |
| 8 | `efficientnet` | EfficientNet: Rethinking Model Scaling for CNNs | 1905.11946 | **완료** |
| 9 | `swin` | Swin Transformer | 2103.14030 | **완료** |
| 10 | `mask-rcnn` | Mask R-CNN | 1703.06870 | **완료** |
| 11 | `ssd` | SSD: Single Shot MultiBox Detector | 1512.02325 | **완료** |

## 우선순위 2 — 손볼 곳이 명확한 숨긴 글 (4편)  ✅ 전부 완료

| # | slug | 문제 | arXiv | 상태 |
|---|---|---|---|---|
| 12 | `v-jepa` | related work 가 원문 직역. `Vidusal encoder-decoder`, `미세조장할 때` 같은 오타. 마지막 문장이 `EMA로 업데이` 에서 끊김 | 2404.08471 | **완료** |
| 13 | `dinov1` | 「해결책」에 `centering`, `sharpness` 두 단어만 있고 설명이 없음 | 2104.14294 | **완료** |
| 14 | `attention-is-all-you-need` | 본문에 `20250319 수정 필요`, `거쳐서 ~~~~` 가 남아 있음. 그림 15장 | 1706.03762 | **완료** |
| 15 | `rlhf` | 논문 4편이 한 글에 뭉쳐 있음 → **분리 필요** (아래 참조) | — | **완료 (4편 분리)** |

### `rlhf` 분리 계획

한 글을 네 편으로 나눈다. 원본의 해당 대목을 각각 옮기고 형식을 맞춘다.

| 새 slug | 논문 | arXiv |
|---|---|---|
| `ppo` | Proximal Policy Optimization Algorithms | 1707.06347 |
| `learning-to-summarize` | Learning to summarize from human feedback | 2009.01325 |
| `instructgpt` | Training language models to follow instructions with human feedback | 2203.02155 |
| `webgpt` | WebGPT: Browser-assisted question-answering with human feedback | 2112.09332 |

원본에 `Table 1 내용 확인`, `이미지 (b) 내용 확인` 같은 미완성 표시가 있다.
논문을 받아서 채우거나, 못 채우면 그 대목을 지운다.

## 우선순위 3 — 나머지 숨긴 글 (10편)

| # | slug | arXiv | 상태 |
|---|---|---|---|
| 16 | `dinov3` | 2508.10104 | **완료** |
| 17 | `v-jepa-2` | 2506.09985 | **완료** |
| 18 | `bert` | 1810.04805 | 대기 |
| 19 | `sam` | 2304.02643 | 대기 |
| 20 | `resnet` | 1512.03385 | 대기 |
| 21 | `unet` | 1505.04597 | 대기 |
| 22 | `swav` | 2006.09882 | 대기 |
| 23 | `ae-vae` | 1312.6114 | 대기 |
| 24 | `simsiam` | 2011.10566 | 대기 |
| 25 | `byol` | 2006.07733 | 대기 |

`simsiam` 과 `byol` 은 분량이 가장 짧다(각각 1,191자 / 883자). 논문에서 보강이 많이 필요하다.
`simsiam` 원본의 *"gradient를 그대로 쓰면 업데이트가 너무 빨라지지 않나?"* 라는 의문은
사용자 본인의 것이므로 **반드시 살린다.**

---

## 상태 표기

- `대기` — 아직 안 함
- `진행중` — 작업 시작함 (중단되면 여기서 이어받는다)
- `완료` — `revisions/reviews/<slug>.md` 생성됨
- `검토됨` — 사용자가 확인함

## 작업 기록

| 날짜 | slug | 한 일 |
|---|---|---|
| 2026-09-10 | — | 형식 명세와 진행표 작성. `vla` frontmatter 에 OpenVLA 링크·서지 확인해 반영 |
| 2026-09-10 | `vla` | 배경 지식·세부 아키텍처·학습 설정·실험 결과 신설. 논문 3.1~4절에서 확인. 끝맺음 평서형 통일 |
| 2026-09-10 | `personaplex` | 배경 지식·실험 결과(Table 1·2 수치) 신설. 절 번호를 내용 제목으로 교체 |
| 2026-09-10 | `dinov2` | 끊긴 마지막 문장(distill)을 논문 5절로 완성. 데이터 구축 수치·효율화 절 신설 |
| 2026-09-10 | `dit` | 조건 주입 4가지 비교·스케일링 절 신설. 끊긴 문장 완성. FID 2.27 등 수치 보강 |
| 2026-09-10 | `ibot` | 형식 재배치. 뒤엉킨 문장 분리, prediction ratio·linear probing 82.3% 보강 |
| 2026-09-10 | `clip` | 「왜 contrastive 인가」(3배·4배 효율 근거), prompt engineering 절 신설 |
| 2026-09-10 | `efficientnet` | α·β²·γ²≈2 제약의 근거(FLOPs 비대칭) 보강. B0 구조표·결과 수치 추가. 고찰 유지 |
| 2026-09-10 | `swin` | 복잡도 식을 텍스트로 명시. cyclic vs padding 비교 추가. 결과를 실제 수치로 교체 |
| 2026-09-10 | `mask-rcnn` | RoIAlign 효과(상대 10~50%)·COCO 결과표 보강. RoIAlign 전개와 FPN 추적은 유지 |
| 2026-09-10 | `ssd` | VOC2007 결과표(74.3%/59FPS, Faster R-CNN·YOLO 대비) 신설. NMS 예시는 유지 |
| 2026-09-10 | `v-jepa` | related work 전면 재작성(원문 직역·오타 제거). 끊긴 문장 완성. frozen 결과표 신설 |
| 2026-09-10 | `dinov1` | 비어 있던 centering/sharpening 을 논문 3절로 채움(서로 반대 작용). k-NN 78.3% 등 결과 |
| 2026-09-10 | `attention-is-all-you-need` | 미완성 표시 3곳을 encoder 블록 구조(FFN·residual·LayerNorm)로 채움. BLEU 결과 신설 |
| 2026-09-10 | `rlhf` | ppo / learning-to-summarize / instructgpt / webgpt 4편으로 분리. Table 1 TODO 를 논문 명령어표로 채움 |
| 2026-09-10 | `dinov3` | 정리 안 된 말미 메모(중복 3회) 재작성. 메모형 단문을 문장으로 |
| 2026-09-10 | `v-jepa-2` | 구어체 도입부 재작성. 데이터 규모 대비(100만 시간 vs 62시간)·zero-shot 배포 결과 보강 |
