# chb2066.github.io

Astro 기반 개인 홈페이지 + 논문 리뷰 블로그.

---

## 1. 최초 1회 세팅

### 1-1. 준비물

Node.js 18 이상. 확인:

```bash
node -v
```

없으면 https://nodejs.org 에서 LTS 설치.

### 1-2. 기존 저장소 정리

`chb2066.github.io` 저장소에 이미 옛날 `index.html`이 있습니다.
그 파일들은 이제 안 쓰이니 지우고 이 프로젝트 파일로 교체하세요.

```bash
git clone https://github.com/chb2066/chb2066.github.io.git
cd chb2066.github.io

# 기존 파일 제거 (.git 폴더는 절대 지우지 말 것)
git rm -r --cached .
rm -rf index.html notes assets   # 있는 것만

# 압축 푼 web/ 안의 내용물을 이 폴더에 복사
```

### 1-3. 설치와 로컬 확인

```bash
npm install
npm run dev
```

http://localhost:4321 로 접속. 파일을 고치면 새로고침 없이 바로 반영됩니다.
끝낼 때 `Ctrl + C`.

### 1-4. 첫 배포

```bash
git add .
git commit -m "Astro 사이트로 전환"
git push
```

그 다음 **GitHub 저장소 → Settings → Pages → Build and deployment →
Source 를 `GitHub Actions` 로 변경**. (기본값 `Deploy from a branch` 로는 동작하지 않습니다.)

저장소 **Actions** 탭에서 초록 체크가 뜨면 배포 완료.
1~2분 뒤 https://chb2066.github.io 에서 확인.

이후로는 `git push` 만 하면 자동 배포됩니다. 로컬 빌드 불필요.

---

## 2. 글 쓰기

### 2-1. 논문 리뷰

`src/content/reviews/_TEMPLATE.md` 를 복사해서 새 이름으로 저장합니다.
**파일 이름이 곧 주소**가 됩니다. `dinov3.md` → `/reviews/dinov3/`
띄어쓰기·한글 대신 영문 소문자와 하이픈을 쓰세요.

```markdown
---
title: DINOv3
paper: "DINOv3: Self-supervised Vision Features at Scale"
venue: arXiv 2025
authors: Meta AI Research
link: https://arxiv.org/abs/2508.10104
claim: 대규모 self-supervised 학습만으로 dense prediction까지 커버하는 범용 시각 표현을 얻는다.
take: frozen 상태로 가장 잘 통했다. 다만 도메인이 멀면 "범용"의 한계가 뚜렷하다.
tags: [Self-supervised, Vision Backbone]
tier: main
date: 2026-08-20
draft: false
---

## 왜 읽었나
...
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `title` | O | 카드에 크게 뜨는 이름. 짧게 |
| `paper` | | 논문 정식 제목. `title`과 같으면 생략 |
| `venue` | | `CVPR 2024` 등. 카드 우측에 표시 |
| `authors` | | 상세 페이지 메타에만 |
| `link` | | arXiv 등 원문 주소 |
| `claim` | O | **논문의** 핵심 주장 한 줄. 카드 본문 |
| `take` | | **내** 판단 한 줄. 카드 하단 강조 영역 |
| `tags` | | 대괄호 배열. 태그 페이지가 자동 생성됨 |
| `tier` | | `main`=본문·판단까지 노출 / `basic`=한 줄 목록 |
| `date` | O | **정렬용. 화면에는 표시되지 않음** |
| `draft` | | `true` 면 빌드에서 제외 |

`title`에 콜론(`:`)이 들어가면 `"큰따옴표"`로 감싸야 합니다. YAML 문법 오류의 90%가 이것입니다.

### 2-2. tier 고르는 기준

- `main` — 분야 관련해서 깊게 본 논문. `claim`과 `take`가 목록에 모두 노출
- `basic` — 이미 아는 기초 논문. 제목과 `claim` 한 줄만. 본문은 짧아도 됨

### 2-3. Knowledge Base / Project Notes

- Knowledge Base → `src/content/knowledge/` — `title`, `summary`, `tags`, `date`
- Project Notes → `src/content/projects/` — `title`, `summary`, `context`, `period`, `role`, `stack`, `tags`, `date`

### 2-3-1. Knowledge Base 작성 규칙

기존 19편이 하나의 형식을 따르고 있습니다. 새 글도 같은 형식으로 써야
목록이 일관되게 읽힙니다.

**한 글 = 한 계열.** 기법 하나에 페이지 하나를 만들지 않습니다.
표현학습, 어텐션, 3D처럼 계열 단위로 묶고 그 안에 기법을 `##` 로 나열합니다.

**글 머리에는 그 계열을 관통하는 한 문장**을 인용문으로 답니다.
"이 계열의 기법들이 공통으로 하는 일이 무엇인가"에 대한 답입니다.

**각 기법은 세 칸으로 적습니다.**

```markdown
## Contrastive Learning / InfoNCE

**하는 일** — 이름 말고 실제로 무슨 연산을 하는가. 한 문장.

**옮길 수 있는 성질**

1. 다른 문제로 가져갈 수 있는 부분.
2. 여러 개면 번호를 매긴다.

**깨지는 지점** — 그 성질이 성립하려면 무엇이 참이어야 하는가.
가정이 깨지는 자리가 이 글에서 가장 중요한 부분입니다.

**비용** — 데이터·연산·추가 학습 요구. 특기할 게 없으면 생략.
```

**글 끝에는 「효과로 다시 보기」 표**를 답니다. 같은 기법들을
"무슨 일을 해주는가"로 다시 묶은 역색인입니다. 문제를 만났을 때
이쪽에서 찾는 게 빠르기 때문에, 이 표가 사실상 글의 색인 역할을 합니다.

| 필요한 것 | 해당 기법 |
|---|---|
| 음성 샘플 없이 학습 | BYOL/SimSiam, DINO, MAE |

### 2-3-2. ⚠️ 전문용어는 번역하지 않습니다

**AI 용어는 영문 원어를 그대로 씁니다.** 번역하면 오히려 못 알아봅니다.

| 이렇게 | 이렇게 쓰지 않습니다 |
|---|---|
| Adversarial Filtering | 적대적 필터링 |
| Conformal Prediction | 등각 예측 |
| Action Chunking | 행동 청킹 |
| stop-gradient | 정지기울기 |
| Hindsight Experience Replay | 사후 경험 재라벨링 |
| proprioception | 고유수용감각 |

- `##` 헤딩은 원어 기법명으로 (`## Domain Randomization`)
- 본문에서도 용어는 원어로 (`contrastive learning의 목적함수는…`)
- 다만 **서술형 섹션 제목은 한국어로** 둡니다
  (`## 효과로 다시 보기`, `### GRPO의 숨은 가정`)
- 연결하는 문장은 당연히 한국어입니다. 용어만 원어입니다.

### 2-4. 이미지

`public/` 에 넣고 절대 경로로 참조합니다.

```markdown
![그림 설명](/img/dinov3-fig1.png)
```

### 2-5. 수식

```markdown
인라인은 $F_1 = 2PR/(P+R)$ 이렇게.

$$
\mathcal{L} = -\sum_i y_i \log \hat{y}_i
$$
```

KaTeX로 렌더링됩니다.

### 2-6. 표와 코드

````markdown
| 모델 | F1@5 |
|---|---|
| DINOv3 | 0.592 |

```python
loss = criterion(logits, targets)
```
````

---

## 3. 커스터마이즈

**색** — `src/styles/global.css` 상단 `:root` 의 `--brand` 한 줄. 다크모드는 `[data-theme="dark"]` 블록의 같은 변수. 회색조 전체를 바꾸려면 `--background` / `--foreground` / `--muted` / `--border`.

**홈 내용** — `src/pages/index.astro`. 히어로 문구, About, 학력, 논문, 프로젝트, 수상이
전부 이 파일 안에 평범한 HTML로 들어 있습니다.

**상단 메뉴** — `src/components/TopBar.astro` 의 `links` 배열.

**사이트 문구는 전부 영어**입니다. 글 본문은 한국어로 써도 되지만 섞으면 어색해지니 한쪽으로 통일하세요.

**좌측 사이드바** — `src/components/Sidebar.astro` 의 `groups` 배열.

**푸터** — `src/layouts/Base.astro` 하단.

**⌘K 검색** — `src/components/CommandPalette.astro`. 모든 글이 자동으로 색인됩니다.

---

## 4. 자주 막히는 것

**빌드가 깨진다** → 대부분 frontmatter YAML 오류입니다. 에러 메시지에 파일 이름이
나오니 그 파일의 `---` 사이를 확인하세요. 콜론, 들여쓰기, 대괄호가 범인입니다.

**Actions에서 빨간 X** → Actions 탭 → 실패한 실행 클릭 → 로그 확인.
`npm ci` 단계에서 실패하면 `package-lock.json`을 커밋했는지 확인하세요.

**페이지가 404** → Settings → Pages의 Source가 `GitHub Actions`인지 확인.

**글을 썼는데 안 보인다** → `draft: true` 로 두지 않았는지, `date`가 있는지 확인.

**로컬은 되는데 배포본에서 링크가 깨진다** → 경로를 `/reviews/` 처럼 슬래시로 시작하고
끝에도 슬래시를 붙이세요.

---

## 5. 폴더 구조

```
src/
  content/
    reviews/     Paper Reviews (_TEMPLATE.md 복사해서 사용)
    knowledge/   Knowledge Base
    projects/    Project Notes
  pages/         라우트 (URL 구조)
  layouts/       Base / Doc(사이드바) / Entry(글 상세)
  components/    TopBar, Sidebar, CommandPalette
  styles/
    global.css   디자인 토큰과 스타일 전부 여기
  content.config.ts   frontmatter 스키마 정의
.github/workflows/deploy.yml   자동 배포
```

새 필드를 추가하고 싶으면 `content.config.ts` 의 스키마에 먼저 넣어야 합니다.
넣지 않은 필드를 md에 쓰면 빌드가 거부합니다. (오타 방지 장치입니다.)
