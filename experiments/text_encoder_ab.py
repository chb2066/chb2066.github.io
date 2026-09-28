# -*- coding: utf-8 -*-
"""텍스트 인코더만 바꿔가며 같은 파이프라인에서 비교.

이미지 feature(동결 SigLIP2 vision), head, split, seed, LR, epoch 을 전부 고정하고
텍스트 인코더만 교체한다. 그래야 '인코더 차이' 와 '파이프라인 차이' 가 안 섞인다.

원본 주석:
SigLIP2 이미지/텍스트 인코더 동결 조건 비교.

물음: 텍스트 타워만 풀면 어떻게 되나, 둘 다 풀면 어떻게 되나.

세 조건을 같은 split, 같은 head, 같은 seed 로 돌린다.
  A  both_frozen    이미지 frozen, 텍스트 frozen
  B  text_only      이미지 frozen, 텍스트 unfrozen
  C  both_unfrozen  이미지 unfrozen, 텍스트 unfrozen

동결된 vision tower 는 출력이 항상 같으므로 experiments/cache/vfeat.npy 를 그대로 쓴다.
푸는 조건에서만 pixels.npy 에서 픽셀을 읽어 tower 를 태운다.

주의. SigLIP2 텍스트 타워는 position embedding 이 64개뿐이라 이 데이터의
description(중앙값 141토큰)을 대부분 잘라낸다. 세 조건 모두 동일하게 64토큰으로
자르므로 조건 간 비교는 유효하지만, 절대값을 512토큰을 쓰는 klue-roberta 계열과
견주면 안 된다.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor

REPO_TRAIN = os.environ.get(
    "KPEC_TRAIN_DIR",
    r"C:/Users/chb07/AppData/Local/Temp/claude/d--CV/36c80b59-0936-4007-8893-360d6ad9b884/scratchpad/repos/korean-pattern-emotion-classification/train",
)
sys.path.insert(0, REPO_TRAIN)
import dataset as D  # noqa: E402  레포의 split·redact 를 그대로 쓴다

MODEL = "google/siglip2-base-patch16-224"
CACHE = Path("experiments/cache")
SEED = 42
MAX_TOK = 64


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Bank:
    """캐시를 기록 id 로 찾아 쓰게 묶어둔다."""

    def __init__(self, proc):
        ids = np.load(CACHE / "ids.npy", allow_pickle=True)
        self.idx = {str(k): i for i, k in enumerate(ids)}
        self.pixels = np.load(CACHE / "pixels.npy", mmap_mode="r")
        self.vfeat = np.load(CACHE / "vfeat.npy")
        self.mean = np.array(proc.image_processor.image_mean, dtype=np.float32)
        self.std = np.array(proc.image_processor.image_std, dtype=np.float32)

    def pix(self, rid):
        a = np.asarray(self.pixels[self.idx[rid]], dtype=np.float32) / 255.0
        return torch.from_numpy(((a - self.mean) / self.std)).permute(2, 0, 1)

    def feat(self, rid):
        return torch.from_numpy(self.vfeat[self.idx[rid]])


class PatternSet(Dataset):
    def __init__(self, recs, vocab, bank, need_pixels):
        self.recs, self.vocab, self.bank = recs, vocab, bank
        self.need_pixels = need_pixels
        self.l2i = {l: i for i, l in enumerate(vocab)}

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = self.bank.pix(r["id"]) if self.need_pixels else self.bank.feat(r["id"])
        txt = D.redact(r["description"], self.vocab)
        y = torch.zeros(len(self.vocab))
        for e in r["emotions"]:
            y[self.l2i[e]] = 1.0
        return img, txt, y


class Collate:
    """윈도우 DataLoader worker 가 pickle 할 수 있게 클래스로 둔다.

    chunks>1 이면 64토큰 창을 여러 개 만들어 (B, chunks, 64) 로 넘긴다.
    SigLIP2 텍스트 타워는 position embedding 이 64개뿐이라 긴 글을 그냥 자르는데,
    창을 나눠 각각 태우고 평균내면 '길이 제약'만 따로 떼어 볼 수 있다.
    """

    def __init__(self, tok, chunks=1, win=MAX_TOK):
        self.tok, self.chunks, self.win = tok, chunks, win

    def __call__(self, batch):
        img = torch.stack([b[0] for b in batch])
        enc = self.tok([b[1] for b in batch], padding="max_length", truncation=True,
                       max_length=self.win * self.chunks, return_tensors="pt")
        ids = enc["input_ids"]
        # SigLIP2 토크나이저는 attention_mask 를 안 내놓는다. 없으면 pad 로 만든다.
        if "attention_mask" in enc:
            mask = enc["attention_mask"]
        else:
            pad = self.tok.pad_token_id
            mask = torch.ones_like(ids) if pad is None else (ids != pad).long()
        if self.chunks > 1:
            ids = ids.view(ids.size(0), self.chunks, self.win)
            mask = mask.view(mask.size(0), self.chunks, self.win)
        y = torch.stack([b[2] for b in batch])
        return img, ids, mask, y


class FusionHead(nn.Module):
    """이미지·텍스트 pooled 벡터를 2토큰 시퀀스로 보고 self-attention 한 번."""

    def __init__(self, dim_i, dim_t, d=512, n_cls=22, p=0.1):
        super().__init__()
        self.pi, self.pt = nn.Linear(dim_i, d), nn.Linear(dim_t, d)
        self.attn = nn.MultiheadAttention(d, 8, dropout=p, batch_first=True)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(p)
        self.out = nn.Linear(2 * d, n_cls)

    def forward(self, vi, vt):
        seq = torch.stack([self.pi(vi), self.pt(vt)], dim=1)
        a, _ = self.attn(seq, seq, seq)
        a = self.norm(seq + a)
        return self.out(self.drop(a.reshape(a.size(0), -1)))


class Net(nn.Module):
    def __init__(self, base, n_cls, train_vision, train_text, text_tower=None, dim_t=None):
        super().__init__()
        self.train_vision, self.train_text = train_vision, train_text
        self.vision = base.vision_model if train_vision else None
        self.text = text_tower if text_tower is not None else base.text_model
        for p in self.text.parameters():
            p.requires_grad = train_text
        if self.vision is not None:
            for p in self.vision.parameters():
                p.requires_grad = True
        self.head = FusionHead(base.config.vision_config.hidden_size,
                               dim_t or base.config.text_config.hidden_size, n_cls=n_cls)

    def train(self, mode=True):
        """동결한 타워는 항상 eval 로 둔다.

        기본 train() 은 자식 모듈을 모두 train 모드로 돌려서 얼린 타워의 dropout 까지
        켜버린다. 그러면 '동결' 조건의 feature 가 스텝마다 흔들려 부당하게 불리해진다.
        """
        super().train(mode)
        if not self.train_text:
            self.text.eval()
        return self

    def _pool(self, ids, mask):
        o = self.text(input_ids=ids, attention_mask=mask)
        p = getattr(o, "pooler_output", None)
        if p is not None:
            return p
        h = o.last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def _text(self, ids, mask):
        """ids 가 (B, chunks, L) 면 창별로 태운 뒤 평균낸다."""
        if ids.dim() == 3:
            b, k, t = ids.shape
            out = self._pool(ids.reshape(b * k, t), mask.reshape(b * k, t))
            return out.view(b, k, -1).mean(1)
        return self._pool(ids, mask)

    def forward(self, img, ids, mask):
        vi = self.vision(pixel_values=img).pooler_output if self.vision is not None else img
        if self.train_text:
            vt = self._text(ids, mask)
        else:
            with torch.no_grad():
                vt = self._text(ids, mask)
        return self.head(vi, vt)


def f1_at5(logits, y):
    hit = y.gather(1, logits.topk(5, dim=1).indices).sum(1)
    return (hit / 5.0).mean().item()


@torch.no_grad()
def evaluate(net, loader, dev, n_cls):
    net.eval()
    tot = n = 0
    hit = torch.zeros(n_cls); gt = torch.zeros(n_cls); pred = torch.zeros(n_cls)
    for img, ids, mask, y in loader:
        img, ids, mask = img.to(dev, non_blocking=True), ids.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(img, ids, mask).float().cpu()
        tot += f1_at5(lg, y) * y.size(0); n += y.size(0)
        top5 = lg.topk(5, dim=1).indices
        for b in range(y.size(0)):
            for c in top5[b]:
                pred[c] += 1
                if y[b, c] > 0:
                    hit[c] += 1
        gt += y.sum(0)
    return tot / n, (2 * hit / (gt + pred).clamp(min=1)).numpy()


def run(cond, train_vision, train_text, vocab, tr, va, args, bank, proc):
    set_seed()
    dev = "cuda"
    base = AutoModel.from_pretrained(MODEL)
    tower, dim_t, tok = None, None, proc.tokenizer
    if args.text_model:
        from transformers import AutoTokenizer
        tower = AutoModel.from_pretrained(args.text_model)
        dim_t = tower.config.hidden_size
        tok = AutoTokenizer.from_pretrained(args.text_model)
    net = Net(base, len(vocab), train_vision, train_text, tower, dim_t).to(dev)

    coll = Collate(tok, args.chunks, args.win)

    def mk(rs, sh):
        return DataLoader(PatternSet(rs, vocab, bank, train_vision), batch_size=args.bs,
                          shuffle=sh, num_workers=args.workers, collate_fn=coll,
                          pin_memory=True, persistent_workers=args.workers > 0)

    dl_tr, dl_va = mk(tr, True), mk(va, False)

    groups = [{"params": net.head.parameters(), "lr": args.head_lr}]
    if train_vision:
        groups.append({"params": net.vision.parameters(), "lr": args.backbone_lr})
    if train_text:
        groups.append({"params": net.text.parameters(), "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups],
        total_steps=args.epochs * len(dl_tr), pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()

    ntr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("[%s] vision=%s text=%s | chunks %d (%d토큰) | trainable %.1fM"
          % (cond, "train" if train_vision else "frozen",
             "train" if train_text else "frozen", args.chunks,
             args.win * args.chunks, ntr / 1e6), flush=True)

    best, best_f1c, hist, t0 = 0.0, None, [], time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        rl = 0.0
        for img, ids, mask, y in dl_tr:
            img, ids, mask, y = (img.to(dev, non_blocking=True), ids.to(dev, non_blocking=True),
                                 mask.to(dev, non_blocking=True), y.to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(net(img, ids, mask).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step()
            rl += loss.item() * y.size(0)
        f1, f1c = evaluate(net, dl_va, dev, len(vocab))
        hist.append(round(f1, 4))
        if f1 > best:
            best, best_f1c = f1, f1c
        print("   ep%-2d loss %.4f  val F1@5 %.4f  (best %.4f)  %.0fs"
              % (ep, rl / len(tr), f1, best, time.time() - t0), flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    del net, base, opt, dl_tr, dl_va
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return {"condition": cond,
            "vision": "train" if train_vision else "frozen",
            "text": "train" if train_text else "frozen",
            "best_f1": round(best, 4), "last_f1": hist[-1], "history": hist,
            "trainable_M": round(ntr / 1e6, 1), "peak_gpu_GB": round(peak, 2),
            "minutes": round((time.time() - t0) / 60, 1),
            "per_label_f1": {vocab[i]: round(float(best_f1c[i]), 4) for i in range(len(vocab))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--text_model", default="",
                    help="비우면 SigLIP2 자체 텍스트 타워. 예: klue/roberta-large")
    ap.add_argument("--win", type=int, default=MAX_TOK, help="창 하나의 토큰 수")
    ap.add_argument("--chunks", type=int, default=1,
                    help="64토큰 창 개수. 1이면 잘라 쓰고, 3이면 192토큰을 세 창으로 나눠 평균")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="experiments/siglip2_freeze_results.json")
    args = ap.parse_args()

    recs = D.load_records()
    vocab = D.build_vocab(recs)
    tr, va = D.split_records(recs)
    print("records %d | train %d | val %d | vocab %d"
          % (len(recs), len(tr), len(va), len(vocab)), flush=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    bank = Bank(proc)

    conds = [("A_both_frozen", False, False),
             ("B_text_only", False, True),
             ("C_both_unfrozen", True, True)]
    if args.only:
        keep = set(args.only.split(","))
        conds = [c for c in conds if c[0] in keep]

    out = []
    for cond, tv, tt in conds:
        out.append(run(cond, tv, tt, vocab, tr, va, args, bank, proc))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model": MODEL, "seed": SEED, "epochs": args.epochs,
                   "batch_size": args.bs, "head_lr": args.head_lr,
                   "backbone_lr": args.backbone_lr, "text_model": args.text_model or MODEL, "win": args.win,
                   "text_max_tokens": args.win * args.chunks, "chunks": args.chunks,
                   "split": {"train": len(tr), "val": len(va)}, "runs": out},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n%-18s %-8s %-8s %8s %8s %7s %6s"
          % ("조건", "vision", "text", "best", "last", "GB", "min"))
    for r in out:
        print("%-18s %-8s %-8s %8.4f %8.4f %7.2f %6.1f"
              % (r["condition"], r["vision"], r["text"], r["best_f1"],
                 r["last_f1"], r["peak_gpu_GB"], r["minutes"]))


if __name__ == "__main__":
    main()
