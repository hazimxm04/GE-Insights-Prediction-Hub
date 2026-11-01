"""
Collect a targeted pool of Malaysian political headlines for labelling (batch 2+).

WHY: batch 1 (110 headlines) added only 1 islam_threat, 0 cost_living, 1 corruption and
2 govt_performance positives, because the pool was almost all election-night coverage.
This script searches Google News RSS with theme-oriented queries, over a wider date range, and
builds a blind labelling sheet.

Run from the project root:
  python finetune/collect_headlines.py                      # default: last 120 days, 40 per query
  python finetune/collect_headlines.py --days 180 --per-query 60 --batch 2

Writes (never touches eval_set.csv):
  eval/label_batch<N>.csv       the sheet to fill (shuffled, blank human_* columns)
  eval/label_batch<N>_key.csv   which query found each headline (do not open until you finish)

Uses only the standard library. Low volume (1 request/second, ~25 requests).
"""
import argparse
import csv
import glob
import html
import os
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

THEMES = ["islam_threat", "malay_unity", "cost_living", "corruption", "govt_performance"]
LABELS = ["human_" + t for t in THEMES]

# (query, theme it is meant to surface). Candidates only: you still label blind.
QUERIES = [
    ("cost of living Malaysia", "cost_living"),
    ("Malaysia prices inflation households", "cost_living"),
    ("subsidi petrol harga barang naik", "cost_living"),
    ("Malaysia subsidy rationalisation RON95", "cost_living"),
    ("Malaysia wages salary minimum wage", "cost_living"),
    ("MACC Malaysia bribery", "corruption"),
    ("rasuah SPRM politik", "corruption"),
    ("Malaysia corruption politician scandal", "corruption"),
    ("Malaysia money politics vote buying", "corruption"),
    ("Islam Malaysia politics threat", "islam_threat"),
    ("3R race religion royalty Malaysia", "islam_threat"),
    ("hudud Malaysia", "islam_threat"),
    ("Islam terancam politik Malaysia", "islam_threat"),
    ("Malaysia non-Muslims fear Islamisation", "islam_threat"),
    ("Anwar government delivery promises", "govt_performance"),
    ("unity government performance Malaysia", "govt_performance"),
    ("Madani economy Malaysia reforms", "govt_performance"),
    ("prestasi kerajaan Madani", "govt_performance"),
    ("Malaysia government failed to deliver", "govt_performance"),
    ("Umno PAS cooperation", "malay_unity"),
    ("Melaka state election", "malay_unity"),
    ("Selangor politics Perikatan Barisan", "malay_unity"),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; headline labelling)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def rss_url(query, days):
    q = urllib.parse.quote_plus(f"{query} when:{days}d")
    return f"https://news.google.com/rss/search?q={q}&hl=en-MY&gl=MY&ceid=MY:en"


def parse_items(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.iter("item"):
        title = html.unescape((it.findtext("title") or "").strip())
        pub = (it.findtext("pubDate") or "").strip()
        src_el = it.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        if title:
            out.append({"title": title, "published": pub, "source": f"GoogleNews-{source}" if source else "GoogleNews"})
    return out


def norm(title):
    t = re.sub(r"\s-\s[^-]+$", "", str(title).lower())
    return re.sub(r"\s+", " ", t).strip()


def known_titles():
    seen = set()
    paths = ["eval/eval_set.csv", "data/processed/scored_articles.csv"] + glob.glob("eval/label_batch*.csv")
    for p in paths:
        if os.path.exists(p) and not p.endswith("_key.csv"):
            with open(p, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("title"):
                        seen.add(norm(row["title"]))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--per-query", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()

    seen = known_titles()
    print(f"{len(seen)} headlines already labelled/scored (will be skipped)")
    rows = []
    for q, theme in QUERIES:
        try:
            items = parse_items(fetch(rss_url(q, a.days)))
        except Exception as e:  # network error, blocked, bad XML
            print(f"  ! {q!r}: {type(e).__name__}: {e}")
            continue
        added = 0
        for it in items[: a.per_query]:
            k = norm(it["title"])
            if k in seen:
                continue
            seen.add(k)
            it.update(query=q, query_theme=theme)
            rows.append(it)
            added += 1
        print(f"  {q!r:55} -> {len(items):3d} found, {added:3d} new")
        time.sleep(a.sleep)

    if not rows:
        print("Nothing collected (network blocked or no new headlines).")
        return
    random.Random(a.seed).shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = i

    sheet = f"eval/label_batch{a.batch}.csv"
    key = f"eval/label_batch{a.batch}_key.csv"
    os.makedirs("eval", exist_ok=True)
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "source", "published"] + LABELS + ["notes"])
        for r in rows:
            w.writerow([r["id"], r["title"], r["source"], r["published"]] + [""] * len(LABELS) + [""])
    with open(key, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "query_theme"])
        for r in rows:
            w.writerow([r["id"], r["query"], r["query_theme"]])
    print(f"\nWrote {len(rows)} headlines to {sheet} (shuffled, blank labels)")
    print(f"Key (which query found each): {key}  <- do not open until you have labelled")


if __name__ == "__main__":
    main()