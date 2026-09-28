# -*- coding: utf-8 -*-
"""동결 텍스트 인코더 + 선형 head 만으로 얼마나 나오나.

이미지를 아예 안 쓴다. 인코더는 얼리고 CLS(또는 pooled) 벡터 위에 linear 하나만 올린다.
동결이라 feature 가 고정이므로 한 번만 뽑아두고 head 만 학습한다.

  python experiments/text_probe.py --text_model klue/roberta-large --win 256 --pool cls
  python experiments/text_probe.py --win 64                      # SigLIP2 텍스트 타워
  python experiments/text_probe.py --win 64 --chunks 4           # 창 4개 평균
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

REPO_TRAIN = os.environ.get(
    "KPEC_TRAIN_DIR",
    r"C:/Users/chb07/AppData/Local/Temp/claude/d--CV/36c80b59-0936-4007-8893-360d6ad9b884/scratchpad/repos/korean-pattern-emotion-classification/train",
)
sys.path.insert(0, REPO_TRAIN)
import dataset as D  # noqa: E402

SIGLIP = "google/siglip2-base-patch16-224"
SEED = 42


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def encode(texts, model, tok, win, chunks, pool, dev="cuda", bs=32):
    """동결 인코더로 텍스트 feature 를 뽑는다."""
    out = []
    model.eval().to(dev)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], padding="max_length", truncation=True,
                  max_length=win * chunks, return_tensors="pt")
        ids = enc["input_ids"]
        mask = enc.get("attention_mask")
        if mask is None:
            pad = tok.pad_token_id
            mask = torch.ones_like(ids) if pad is None else (ids != pad).long()
        b = ids.size(0)
        if chunks > 1:
            ids, mask = ids.view(b * chunks, win), mask.view(b * chunks, win)
        ids, mask = ids.to(dev), mask.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(input_ids=ids, attention_mask=mask)
        if pool == "cls":
            v = o.last_hidden_state[:, 0]
        elif pool == "mean":
            h = o.last_hidden_state
            m = mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:                                    # pooler
            p = getattr(o, "pooler_output", None)
            v = p if p is not None else o.last_hidden_state[:, -1]
        v = v.float()
        if chunks > 1:
            v = v.view(b, chunks, -1).mean(1)
        out.append(v.cpu())
    return torch.cat(out)


def f1_at5(logits, y):
    return (y.gather(1, logits.topk(5, 1).indices).sum(1) / 5.0).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_model", default="", help="비우면 SigLIP2 텍스트 타워")
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--pool", default="", help="cls | mean | pooler (기본: roberta=cls, siglip=pooler)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", default="experiments/text_probe_results.json")
    args = ap.parse_args()

    set_seed()
    recs = D.load_records(); vocab = D.build_vocab(recs)
    tr, va = D.split_records(recs)
    l2i = {l: i for i, l in enumerate(vocab)}

    if args.text_model:
        tok = AutoTokenizer.from_pretrained(args.text_model)
        model = AutoModel.from_pretrained(args.text_model)
        pool = args.pool or "cls"
    else:
        tok = AutoProcessor.from_pretrained(SIGLIP).tokenizer
        model = AutoModel.from_pretrained(SIGLIP).text_model
        pool = args.pool or "pooler"

    name = args.text_model or (SIGLIP + " (text tower)")
    print("인코더 %s | %d토큰 x %d창 | pool=%s" % (name, args.win, args.chunks, pool), flush=True)

    def feats(rs):
        txt = [D.redact(r["description"], vocab) for r in rs]
        X = encode(txt, model, tok, args.win, args.chunks, pool)
        Y = torch.zeros(len(rs), len(vocab))
        for i, r in enumerate(rs):
            for e in r["emotions"]:
                Y[i, l2i[e]] = 1.0
        return X, Y

    t0 = time.time()
    Xtr, Ytr = feats(tr)
    Xva, Yva = feats(va)
    del model; torch.cuda.empty_cache()
    print("feature %s  추출 %.0fs" % (tuple(Xtr.shape), time.time() - t0), flush=True)

    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr, Xva = ((Xtr - mu) / sd).cuda(), ((Xva - mu) / sd).cuda()
    Ytr, Yva = Ytr.cuda(), Yva.cuda()

    set_seed()
    head = nn.Linear(Xtr.size(1), len(vocab)).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.BCEWithLogitsLoss()

    best, hist = 0.0, []
    for ep in range(1, args.epochs + 1):
        head.train()
        opt.zero_grad(set_to_none=True)
        loss = lossf(head(Xtr), Ytr)
        loss.backward(); opt.step(); sched.step()
        head.eval()
        with torch.no_grad():
            f1 = f1_at5(head(Xva).cpu(), Yva.cpu())
        hist.append(round(f1, 4)); best = max(best, f1)
        if ep % 25 == 0 or ep == 1:
            print("   ep%-4d loss %.4f  val F1@5 %.4f  (best %.4f)" % (ep, loss.item(), f1, best), flush=True)

    print("\n최고 val F1@5 = %.4f  (%s, %d토큰, pool=%s)" % (best, name, args.win * args.chunks, pool))
    rec = {"text_model": name, "win": args.win, "chunks": args.chunks, "pool": pool,
           "dim": int(Xtr.size(1)), "best_f1": round(best, 4), "last_f1": hist[-1],
           "epochs": args.epochs, "split": {"train": len(tr), "val": len(va)}}
    p = Path(args.out)
    allr = json.load(open(p, encoding="utf-8")) if p.exists() else []
    allr.append(rec)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(allr, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
