"""
5-fold cross-validation for the theme classifier (Session 2).

WHY: evaluate.py scored all 100 articles, but the model had trained on 80 of them,
so 0.291 macro-F1 was inflated. Here every article is scored exactly once by a model
that never saw it ("out-of-fold" predictions), and everything is compared on the
SAME folds.

Compares:
  - TF-IDF + logistic regression   (Lecture 2 baseline)
  - DistilBERT full fine-tune      (class-weighted BCE, same settings as your train.py)
  - Groq                           (only if per-article Groq scores are in the CSV)

Run from the project root:
  python finetune/cross_validate.py --skip-bert      # TF-IDF only, ~10 seconds
  python finetune/cross_validate.py                  # + DistilBERT, ~20-30 min on CPU
  python finetune/cross_validate.py --epochs 5       # try another epoch count

Output: printed table + eval/cv_results.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

EVAL_CSV = "eval/eval_set.csv"
OUT_JSON = "eval/cv_results.json"
MODEL_NAME = "distilbert-base-uncased"
THEMES = ["islam_threat", "malay_unity", "cost_living", "corruption", "govt_performance"]
PW_CAP = None   # set by --pos-weight-cap; None = no cap
THRESHOLDS = (0.5, 0.3)  # 0.3 = the threshold your Groq baseline used


# ---------------------------------------------------------------- metrics
def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def per_theme_f1(Y, P, thr):
    pred = (P >= thr).astype(int)
    return [prf(Y[:, j], pred[:, j])[2] for j in range(Y.shape[1])]


def macro_f1(Y, P, thr):
    # average only over themes that have at least one positive in this sample
    f1s = [f for f, j in zip(per_theme_f1(Y, P, thr), range(Y.shape[1])) if Y[:, j].sum() > 0]
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_ci(Y, P, thr, n_boot=1000, seed=0):
    """Resample ARTICLES with replacement -> 95% interval for macro-F1."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(Y), len(Y))
        vals.append(macro_f1(Y[idx], P[idx], thr))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- models
def tfidf_oof(texts, Y, folds):
    P = np.zeros(Y.shape)
    for tr, te in folds:
        # IMPORTANT: fit the vectorizer (vocabulary + IDF) on the TRAIN part only
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
        Xtr = vec.fit_transform(texts[tr])
        Xte = vec.transform(texts[te])
        for j in range(Y.shape[1]):
            ytr = Y[tr, j]
            if ytr.sum() == 0:            # theme never appears in this train fold
                P[te, j] = 0.0
            elif ytr.sum() == len(ytr):
                P[te, j] = 1.0
            else:
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
                clf.fit(Xtr, ytr)
                P[te, j] = clf.predict_proba(Xte)[:, 1]
    return P


def bert_oof(texts, Y, folds, epochs, lr, bs, seed):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {dev}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tok(list(texts), padding="max_length", truncation=True, max_length=128,
              return_tensors="pt")
    enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def batch(idx):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        b = {k: v[idx_t] for k, v in enc.items()}
        L = int(b["attention_mask"].sum(1).max())      # trim padding -> faster on CPU
        return {k: v[:, :L].to(dev) for k, v in b.items()}

    P = np.zeros(Y.shape)
    for k, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        torch.manual_seed(seed + k)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(THEMES), problem_type="multi_label_classification"
        ).to(dev)

        # class weights come from the TRAIN part of this fold only
        pos = Y[tr].sum(0)
        w = (len(tr) - pos) / np.maximum(pos, 1)
        if PW_CAP:
            w = np.minimum(w, PW_CAP)      # stops a 3-positive theme getting weight ~100+
        pos_weight = torch.tensor(w, dtype=torch.float32, device=dev)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total = epochs * int(np.ceil(len(tr) / bs))
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

        model.train()
        for ep in range(epochs):
            order = np.random.RandomState(seed * 1000 + k * 100 + ep).permutation(len(tr))
            for i in range(0, len(tr), bs):
                idx = tr[order[i:i + bs]]
                logits = model(**batch(idx)).logits
                target = torch.tensor(Y[idx], dtype=torch.float32, device=dev)
                loss = loss_fn(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

        model.eval()
        with torch.no_grad():
            for i in range(0, len(te), 32):
                idx = te[i:i + 32]
                P[idx] = torch.sigmoid(model(**batch(idx)).logits).cpu().numpy()
        print(f"  fold {k}/{len(folds)} done in {time.time() - t0:.0f}s "
              f"(last-epoch train loss {loss.item():.3f})")
        del model
    return P


def groq_scores(df, has_human=False):
    """Groq's 0-1 score per theme. With human_* labels present, Groq's scores are the bare
    theme columns (islam_threat, ...). Otherwise look for one numeric column per theme."""
    if has_human:
        print(f"  Groq score columns used (bare theme columns): {THEMES}")
        return df[THEMES].values.astype(float)
    cols = []
    for t in THEMES:
        c = [x for x in df.columns
             if t in x and x != t and df[x].dtype.kind in "fi"]
        if len(c) != 1:
            print(f"  (Groq: could not find exactly one score column for '{t}'; "
                  f"candidates: {c}. Skipping Groq.)")
            return None
        cols.append(c[0])
    print(f"  Groq score columns used: {cols}")
    return df[cols].values.astype(float)


def per_theme_ap(Y, P):
    """Average precision per theme = ranking quality, independent of any threshold."""
    return [average_precision_score(Y[:, j], P[:, j]) if Y[:, j].sum() > 0 else float("nan")
            for j in range(Y.shape[1])]


def macro_ap(Y, P):
    aps = [a for a in per_theme_ap(Y, P) if not np.isnan(a)]
    return float(np.mean(aps)) if aps else 0.0


def bootstrap_ap_ci(Y, P, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(Y), len(Y))
        vals.append(macro_ap(Y[idx], P[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff(Y, Pa, Pb, thr, n_boot=2000, seed=1):
    """Same resampled articles for both methods -> is A really better than B?"""
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(Y), len(Y))
        d.append(macro_f1(Y[idx], Pa[idx], thr) - macro_f1(Y[idx], Pb[idx], thr))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--eval-csv", default=EVAL_CSV,
                    help="labelled test set (default eval/eval_set.csv); outputs get a suffix if changed")
    ap.add_argument("--pos-weight-cap", type=float, default=0,
                    help="cap the class weight for rare themes (e.g. 10); 0 = no cap (default)")
    ap.add_argument("--extra", default=None,
                    help="extra hand-labelled CSV (title + human_* columns), used for TRAINING only; "
                         "the scored test articles are still only those in eval_set.csv")
    a = ap.parse_args()
    global PW_CAP
    PW_CAP = a.pos_weight_cap or None

    df = pd.read_csv(a.eval_csv)
    human = ["human_" + t for t in THEMES]
    if all(c in df.columns for c in human):
        label_cols, has_human = human, True       # hand labels = ground truth
    else:
        label_cols, has_human = THEMES, False     # older file layout: bare columns are the labels
    df = df.dropna(subset=label_cols).reset_index(drop=True)
    texts = df["text_snippet"].fillna("").astype(str).values
    Y = df[label_cols].values.astype(int)
    print(f"Loaded {len(df)} labeled articles from {a.eval_csv}")
    print("Ground-truth columns:", label_cols)
    print("Positives per theme:", dict(zip(THEMES, Y.sum(0).tolist())))

    folds = list(KFold(a.folds, shuffle=True, random_state=a.seed).split(texts))
    n = len(texts)
    texts_all, Y_all, folds_tr = texts, Y, folds
    if a.extra:
        ex = pd.read_csv(a.extra).dropna(subset=label_cols).reset_index(drop=True)
        ex_texts = ex["title"].fillna("").astype(str).values
        ex_Y = ex[label_cols].values.astype(int)
        texts_all = np.concatenate([texts, ex_texts]); Y_all = np.vstack([Y, ex_Y])
        extra_idx = np.arange(n, n + len(ex))
        folds_tr = [(np.concatenate([tr, extra_idx]), te) for tr, te in folds]
        print(f"+ {len(ex)} extra training headlines from {a.extra} (never scored)")
        print("Positives in extra:", dict(zip(THEMES, ex_Y.sum(0).tolist())))
    results = {}

    print("\n[1/3] TF-IDF + logistic regression")
    results["tfidf_lr"] = tfidf_oof(texts_all, Y_all, folds_tr)[:n]

    if not a.skip_bert:
        print(f"\n[2/3] DistilBERT ({a.epochs} epochs, lr {a.lr}, batch {a.batch_size})")
        results["distilbert"] = bert_oof(texts_all, Y_all, folds_tr, a.epochs, a.lr, a.batch_size, a.seed)[:n]

    print("\n[3/3] Groq (from CSV, if available)")
    g = groq_scores(df, has_human)
    if g is not None:
        results["groq"] = g      # not trained here, so all articles count
    results["always_yes"] = np.ones(Y.shape)   # dumbest baseline: every theme on every article

    # ---- report
    out = {"n": len(df), "extra": a.extra, "folds": a.folds, "seed": a.seed, "epochs": a.epochs,
           "positives": dict(zip(THEMES, Y.sum(0).tolist())), "methods": {}}
    for thr in THRESHOLDS:
        print(f"\n{'=' * 78}\nthreshold = {thr}   (out-of-fold; each article scored by a model "
              f"that never trained on it)\n{'=' * 78}")
        print(f"{'method':<14}" + "".join(f"{t[:11]:>12}" for t in THEMES) + f"{'MACRO':>9}   95% CI")
        for name, P in results.items():
            f1s = per_theme_f1(Y, P, thr)
            m = macro_f1(Y, P, thr)
            lo, hi = bootstrap_ci(Y, P, thr)
            print(f"{name:<14}" + "".join(f"{f:>12.3f}" for f in f1s) + f"{m:>9.3f}   [{lo:.2f}, {hi:.2f}]")
            out["methods"].setdefault(name, {})[str(thr)] = {
                "per_theme_f1": dict(zip(THEMES, [round(f, 3) for f in f1s])),
                "macro_f1": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)]}

    # ---- how often does each method say "yes"? (true rate shown first)
    print(f"\n{'=' * 78}\nshare of articles each method marks positive (true share first)\n{'=' * 78}")
    print(f"{'':<22}" + "".join(f"{t[:11]:>12}" for t in THEMES))
    print(f"{'TRUE':<22}" + "".join(f"{v:>12.2f}" for v in Y.mean(0)))
    for thr in THRESHOLDS:
        for name, P in results.items():
            if name != "always_yes":
                print(f"{name + ' @' + str(thr):<22}" + "".join(f"{v:>12.2f}" for v in (P >= thr).mean(0)))

    # ---- threshold-free comparison
    print(f"\n{'=' * 78}\naverage precision (threshold-free ranking quality; chance = true share above)\n{'=' * 78}")
    print(f"{'method':<14}" + "".join(f"{t[:11]:>12}" for t in THEMES) + f"{'MACRO':>9}   95% CI")
    for name, P in results.items():
        aps = per_theme_ap(Y, P)
        m = macro_ap(Y, P)
        lo, hi = bootstrap_ap_ci(Y, P)
        print(f"{name:<14}" + "".join(f"{x:>12.3f}" for x in aps) + f"{m:>9.3f}   [{lo:.2f}, {hi:.2f}]")
        out["methods"].setdefault(name, {})["average_precision"] = {
            "per_theme": dict(zip(THEMES, [None if np.isnan(x) else round(x, 3) for x in aps])),
            "macro": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)]}

    # ---- paired comparisons (same resampled articles for both methods)
    pairs = [(x, y) for x, y in [("distilbert", "groq"), ("distilbert", "tfidf_lr"), ("distilbert", "always_yes"),
                                 ("tfidf_lr", "groq"), ("groq", "always_yes")] if x in results and y in results]
    for thr in THRESHOLDS:
        print(f"\n{'=' * 78}\npaired comparison at threshold {thr}: macro-F1 of A minus B "
              f"(95% CI; P = share of resamples where A > B)\n{'=' * 78}")
        for x, y in pairs:
            d, lo, hi, p = paired_diff(Y, results[x], results[y], thr)
            verdict = "A better" if lo > 0 else ("B better" if hi < 0 else "no clear difference")
            print(f"{x + ' - ' + y:<28}{d:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]  P={p:.2f}  -> {verdict}")

    # ---- save out-of-fold predictions so nothing needs retraining to inspect them
    oof = pd.DataFrame({"title": df["title"].values if "title" in df.columns else texts})
    for j, t in enumerate(THEMES):
        oof["true_" + t] = Y[:, j]
        for name, P in results.items():
            if name != "always_yes":
                oof[f"{name}_{t}"] = np.round(P[:, j], 4)
    oof_path = "eval/cv_oof_predictions.csv" if a.eval_csv == EVAL_CSV else "eval/cv_oof_predictions_" + os.path.splitext(os.path.basename(a.eval_csv))[0] + ".csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {oof_path} (every article's out-of-fold probability per method)")

    out_json = OUT_JSON
    if a.eval_csv != EVAL_CSV:
        out_json = OUT_JSON.replace(".json", "_" + os.path.splitext(os.path.basename(a.eval_csv))[0] + ".json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    print("Reading the table: if two methods' 95% CIs overlap heavily, you cannot say one is better.")


if __name__ == "__main__":
    main()