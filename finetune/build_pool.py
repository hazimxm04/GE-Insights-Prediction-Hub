"""
finetune/build_pool.py
======================
Pool every hand-labelled headline into ONE file, so DistilBERT, TF-IDF and Groq can be
compared on the same articles, on all five themes, with cross-validation.

    eval/eval_set_v2.csv   (100, the original test set)
    eval/label_batch1.csv  (110)
    eval/label_batch2.csv  (your new batch)
        |
        v
    eval/pool_v3.csv       title, source, published, text_snippet, human_*, origin

Rules:
  * only rows with all five human_* labels filled (0 or 1) are kept
  * a headline that appears in more than one file is kept ONCE (eval_set_v2 first, then
    batch1, then batch2), so no headline can be in both a training and a test fold
  * duplicates are matched ignoring case and punctuation
  * Groq's old scores are NOT copied: score_pool_groq.py re-scores every headline in one run

Run from the project root:   python finetune/build_pool.py
"""
import argparse
import re
import sys

import pandas as pd

THEMES = ["islam_threat", "malay_unity", "cost_living", "corruption", "govt_performance"]
HUMAN = ["human_" + t for t in THEMES]
KEEP = ["title", "source", "published"]


def norm(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def load(path, origin):
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in ["title"] + HUMAN if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    for c in ("source", "published"):
        if c not in df.columns:
            df[c] = ""
    n_all = len(df)
    df = df.dropna(subset=HUMAN).copy()
    bad = ~df[HUMAN].isin([0, 1, 0.0, 1.0]).all(axis=1)
    if bad.any():
        sys.exit(f"{path}: labels other than 0/1 in rows {df.index[bad].tolist()[:10]}")
    df[HUMAN] = df[HUMAN].astype(int)
    df["origin"] = origin
    print(f"  {origin:<16}{len(df):>4} labelled of {n_all} rows")
    return df[KEEP + HUMAN + ["origin"]]


def build(paths):
    frames = [load(p, o) for p, o in paths]
    pool = pd.concat(frames, ignore_index=True)
    pool["_key"] = pool["title"].map(norm)
    before = len(pool)
    dup = pool[pool.duplicated("_key", keep="first")]
    pool = pool.drop_duplicates("_key", keep="first").drop(columns="_key").reset_index(drop=True)
    if len(dup):
        print(f"  dropped {len(dup)} repeated headline(s): " + "; ".join(dup["title"].str[:50].tolist()))
    pool.insert(3, "text_snippet", pool["title"])      # headline-only, same as the original test set
    print(f"  pool: {len(pool)} headlines (from {before})")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval/eval_set_v2.csv")
    ap.add_argument("--batch1", default="eval/label_batch1.csv")
    ap.add_argument("--batch2", default="eval/label_batch2.csv")
    ap.add_argument("--out", default="eval/pool_v3.csv")
    a = ap.parse_args()
    print("Reading labelled headlines:")
    pool = build([(a.eval, "eval_set_v2"), (a.batch1, "label_batch1"), (a.batch2, "label_batch2")])
    print("\nPositives per theme in the pool:")
    for t in THEMES:
        print(f"  {t:<18}{int(pool['human_' + t].sum()):>4}")
    pool.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {a.out}")
    print("Next: python finetune/score_pool_groq.py")


if __name__ == "__main__":
    main()
