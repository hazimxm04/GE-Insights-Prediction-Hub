"""
finetune/score_pool_groq.py
===========================
Score every pooled headline with the SAME Groq scorer used in production
(sentiment/scoring/sentiment_scorer.py: llama-3.1-8b-instant, same prompt), headline only.

    eval/pool_v3.csv  ->  eval/pool_v3_scored.csv   (+ the five bare theme columns = Groq's scores)

cross_validate.py reads those bare columns as "Groq" and the human_* columns as the truth.

Safe to stop and rerun: finished headlines are kept in eval/pool_v3_groq_cache.csv and skipped.
A call that fails or returns text without the theme lines is retried, and never written
as a "0" (the production scorer would silently turn a failure into all-zero scores).

The production scorer is hard-coded to llama-3.1-8b-instant, which Groq has shut down
(requests return 404 model_not_found). This script therefore overrides the model name for
its own calls (--model, default openai/gpt-oss-20b, Groq's recommended replacement) and
leaves sentiment_scorer.py untouched. The model name is saved with the scores, and each
model has its own cache file, so scores from different models are never mixed.

Run from the project root (needs GROQ_API_KEY in backend/.env):
    python finetune/score_pool_groq.py --limit 5     # quick trial
    python finetune/score_pool_groq.py               # about 12 minutes for ~350 headlines
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.getcwd())
THEMES = ["islam_threat", "malay_unity", "cost_living", "corruption", "govt_performance"]
DEFAULT_MODEL = "openai/gpt-oss-20b"


def load_scorer(model):
    from sentiment.scoring import sentiment_scorer as ss      # same scorer as production
    state = {"text": None, "error": None}
    original = ss.client.chat.completions.create

    def watched(*args, **kwargs):                              # remember the raw reply or the error
        kwargs["model"] = model                                # the production name is retired
        if "gpt-oss" in model:                                 # reasoning models spend tokens thinking:
            kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), 1500)   # leave room for the answer
            kwargs["reasoning_effort"] = "low"
        try:
            r = original(*args, **kwargs)
            state["text"], state["error"] = r.choices[0].message.content or "", None
            return r
        except Exception as e:
            state["text"], state["error"] = None, e
            raise

    ss.client.chat.completions.create = watched
    return ss, state


def score_one(ss, state, text, tries=3):
    for k in range(tries):
        state["text"] = state["error"] = None
        scores = ss.score_article_all(text)
        ok = state["error"] is None and state["text"] and "govt_performance" in state["text"].lower()
        if ok:
            return {t: scores[t] for t in THEMES}
        time.sleep(5 * (k + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="eval/pool_v3.csv")
    ap.add_argument("--out", default="eval/pool_v3_scored.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Groq model id (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (trial run)")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between calls (free tier: 30/min)")
    a = ap.parse_args()

    pool = pd.read_csv(a.pool, encoding="utf-8-sig")
    cache = "eval/pool_v3_groq_cache_" + a.model.replace("/", "_") + ".csv"
    done = pd.read_csv(cache, encoding="utf-8-sig") if os.path.exists(cache) else pd.DataFrame(columns=["title"] + THEMES)
    have = set(done["title"])
    todo = [t for t in pool["title"] if t not in have]
    if a.limit:
        todo = todo[:a.limit]
    print(f"model: {a.model}")
    print(f"{len(pool)} headlines in the pool, {len(have)} already scored, {len(todo)} to score now")

    if todo:
        ss, state = load_scorer(a.model)
        rows, failed = [], []
        for i, title in enumerate(todo, 1):
            s = score_one(ss, state, title)
            if s is None:
                failed.append(title)
                print(f"  [{i}/{len(todo)}] FAILED: {title[:60]}")
            else:
                rows.append({"title": title, **s})
                print(f"  [{i}/{len(todo)}] " + " ".join(f"{t[:5]}={s[t]:.1f}" for t in THEMES) + f"  {title[:45]}")
                pd.concat([done, pd.DataFrame(rows)]).to_csv(cache, index=False, encoding="utf-8-sig")
            time.sleep(a.sleep)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        if failed:
            print(f"\n{len(failed)} headline(s) failed; rerun the script to retry them.")

    scored = pool.merge(done.drop_duplicates("title"), on="title", how="inner")
    scored["groq_model"] = a.model
    missing = len(pool) - len(scored)
    if missing:
        print(f"\n{missing} headline(s) still have no Groq score, so {a.out} is NOT complete. Rerun this script.")
        if a.limit:
            print("(that is expected for a --limit trial)")
    scored.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {a.out} ({len(scored)} of {len(pool)} headlines)")
    if not missing:
        print("Next: python finetune/cross_validate.py --eval-csv eval/pool_v3_scored.csv")


if __name__ == "__main__":
    main()