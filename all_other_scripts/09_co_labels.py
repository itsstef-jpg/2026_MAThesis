
"""
Compute post-level co-labelling statistics for the classified corpus.

The classified-posts file is read line by line and dispatched into five accumulators (label-count distribution, per-regime post count, per-regime engagement weight, pair count, pair engagement weight).
This allows the script to run on the full ~1.5M-post file without
holding more than one post in memory at a time.

"""

import os
import csv
from collections import Counter
from itertools import combinations

try:
    import ujson as json
except ImportError:
    import json

from config import WORK_DIR, REGIME_LABELS

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Paths and output bookkeeping
INPUT_PATH = os.path.join(WORK_DIR, "classified_posts.jsonl")

OUT_DIR = os.path.join(WORK_DIR, "co_labelling")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SUMMARY    = os.path.join(OUT_DIR, "co_labelling_summary.txt")
OUT_PAIRS      = os.path.join(OUT_DIR, "co_labelling_pairs.csv")
OUT_PER_REGIME = os.path.join(OUT_DIR, "co_labelling_per_regime.csv")
OUT_ENGAGEMENT = os.path.join(OUT_DIR, "co_labelling_engagement.csv")

_written = []


def wrote(path):
    _written.append(path)
    print("Wrote:", path)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Streaming pass and output writing

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH!r}.\n"
            "This file is produced by 04_v2_apply_models.py. "
            "Run that script first, or check that WORK_DIR in config.py "
            "points at the directory where it was written."
        )

    # Per-post counters
    total_posts = 0
    posts_by_label_count = Counter()
    regime_post_count = Counter()
    pair_count = Counter()

    # Engagement-weighted counters (parallel to above)
    total_weight = 0.0
    regime_weight = Counter()
    pair_weight = Counter()

    print(f"Reading {INPUT_PATH} ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                # Skip malformed line but report it once
                if i < 5:
                    print(f"  [warn] could not parse line {i}: {e}")
                continue

            regimes_str = rec.get("regimes", "")
            regimes = [r for r in regimes_str.split("|") if r]
            if not regimes:
                continue

            # Engagement weight (likes + reposts + replies + quotes)

            w = float(rec.get("weight", 1.0))

            total_posts += 1
            total_weight += w
            posts_by_label_count[len(regimes)] += 1

            # Single-regime contribution
            for r in regimes:
                regime_post_count[r] += 1
                regime_weight[r] += w

            # Pairwise contribution: every unordered pair of distinct regimes that co-occur on this post
            if len(regimes) > 1:
                for a, b in combinations(sorted(set(regimes)), 2):
                    key = (a, b)
                    pair_count[key] += 1
                    pair_weight[key] += w

            # progress ping every 500k records (the file is large)
            if (i + 1) % 500_000 == 0:
                print(f"  processed {i + 1:,} lines; {total_posts:,} posts so far")

    print(
        f"Done. Total posts: {total_posts:,}, "
        f"total engagement weight: {total_weight:,.0f}"
    )

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Output 1: human-readable summary
    summary_lines = []
    add = summary_lines.append

    add("CO-LABELLING SUMMARY")
    add("=" * 70)
    add(f"Total posts:                   {total_posts:>10,}")
    add(f"Total engagement weight:       {total_weight:>10,.0f}")
    add("")
    add("Posts by number of regime labels:")
    for n in sorted(posts_by_label_count):
        n_posts = posts_by_label_count[n]
        pct = 100.0 * n_posts / total_posts if total_posts else 0.0
        add(f"  {n} label(s): {n_posts:>10,} posts  ({pct:5.2f}%)")

    multi = sum(c for n, c in posts_by_label_count.items() if n > 1)
    multi_pct = 100.0 * multi / total_posts if total_posts else 0.0
    add(f"  Multi-label total: {multi:,} ({multi_pct:.2f}%)")
    add("")

    add("Per-regime totals (posts carrying that label, including in multi-label posts):")
    for r in REGIME_LABELS:
        n = regime_post_count.get(r, 0)
        w = regime_weight.get(r, 0.0)
        add(f"  {r:<46}  posts={n:>9,}   eng_weight={w:>14,.0f}")
    add("")

    # Most common pairs (by count and by engagement weight)
    add("Top regime pairs by post count (raw co-occurrence):")
    for (a, b), n in sorted(pair_count.items(), key=lambda x: -x[1])[:20]:
        a_total = regime_post_count.get(a, 0)
        b_total = regime_post_count.get(b, 0)
        pct_of_a = 100.0 * n / a_total if a_total else 0.0
        pct_of_b = 100.0 * n / b_total if b_total else 0.0
        add(
            f"  {a} ↔ {b}\n"
            f"    co-labelled posts: {n:,}  "
            f"({pct_of_a:.2f}% of {a}; {pct_of_b:.2f}% of {b})"
        )
    add("")

    add("Top regime pairs by engagement weight:")
    for (a, b), w in sorted(pair_weight.items(), key=lambda x: -x[1])[:20]:
        a_w = regime_weight.get(a, 0.0)
        b_w = regime_weight.get(b, 0.0)
        pct_of_a = 100.0 * w / a_w if a_w else 0.0
        pct_of_b = 100.0 * w / b_w if b_w else 0.0
        add(
            f"  {a} ↔ {b}\n"
            f"    co-labelled engagement weight: {w:,.0f}  "
            f"({pct_of_a:.2f}% of {a}'s eng; {pct_of_b:.2f}% of {b}'s eng)"
        )

    summary_text = "\n".join(summary_lines) + "\n"
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)
    wrote(OUT_SUMMARY)

    # ---------------------------------------------------------------------
    # Output 2: every pair, raw counts
    with open(OUT_PAIRS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "regime_a", "regime_b",
            "co_labelled_posts",
            "regime_a_total_posts",
            "regime_b_total_posts",
            "pct_of_regime_a",
            "pct_of_regime_b",
            "co_labelled_engagement_weight",
            "regime_a_total_eng_weight",
            "regime_b_total_eng_weight",
            "pct_of_regime_a_eng",
            "pct_of_regime_b_eng",
        ])
        for a, b in combinations(REGIME_LABELS, 2):
            pair = tuple(sorted([a, b]))
            n = pair_count.get(pair, 0)
            wt = pair_weight.get(pair, 0.0)
            a_n = regime_post_count.get(a, 0)
            b_n = regime_post_count.get(b, 0)
            a_w = regime_weight.get(a, 0.0)
            b_w = regime_weight.get(b, 0.0)
            w.writerow([
                a, b,
                n,
                a_n, b_n,
                f"{100.0 * n / a_n:.4f}" if a_n else "0.0000",
                f"{100.0 * n / b_n:.4f}" if b_n else "0.0000",
                f"{wt:.2f}",
                f"{a_w:.2f}", f"{b_w:.2f}",
                f"{100.0 * wt / a_w:.4f}" if a_w else "0.0000",
                f"{100.0 * wt / b_w:.4f}" if b_w else "0.0000",
            ])
    wrote(OUT_PAIRS)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Output 3: per-regime breakdown, for each regime, what fraction of its posts also carry each other label
    with open(OUT_PER_REGIME, "w", encoding="utf-8", newline="") as f:
        cols = ["regime"] + [f"pct_also_{r}" for r in REGIME_LABELS]
        w = csv.writer(f)
        w.writerow(cols)
        for primary in REGIME_LABELS:
            primary_n = regime_post_count.get(primary, 0)
            row = [primary]
            for other in REGIME_LABELS:
                if other == primary:
                    row.append("")
                    continue
                pair = tuple(sorted([primary, other]))
                n = pair_count.get(pair, 0)
                pct = 100.0 * n / primary_n if primary_n else 0.0
                row.append(f"{pct:.4f}")
            w.writerow(row)
    wrote(OUT_PER_REGIME)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Output 4: same but engagement-weighted
    with open(OUT_ENGAGEMENT, "w", encoding="utf-8", newline="") as f:
        cols = ["regime"] + [f"eng_pct_also_{r}" for r in REGIME_LABELS]
        w = csv.writer(f)
        w.writerow(cols)
        for primary in REGIME_LABELS:
            primary_w = regime_weight.get(primary, 0.0)
            row = [primary]
            for other in REGIME_LABELS:
                if other == primary:
                    row.append("")
                    continue
                pair = tuple(sorted([primary, other]))
                wt = pair_weight.get(pair, 0.0)
                pct = 100.0 * wt / primary_w if primary_w else 0.0
                row.append(f"{pct:.4f}")
            w.writerow(row)
    wrote(OUT_ENGAGEMENT)

    # Print a small console preview of the most useful table.
    print("\nQuick preview of per-regime co-labelling fractions (raw post count):")
    print(f"  {'regime':<46}", end="")
    for r in REGIME_LABELS:
        print(f"{r[:14]:>15}", end="")
    print()
    for primary in REGIME_LABELS:
        primary_n = regime_post_count.get(primary, 0)
        print(f"  {primary:<46}", end="")
        for other in REGIME_LABELS:
            if other == primary:
                print(f"{'—':>15}", end="")
                continue
            pair = tuple(sorted([primary, other]))
            n = pair_count.get(pair, 0)
            pct = 100.0 * n / primary_n if primary_n else 0.0
            print(f"{pct:>14.2f}%", end="")
        print()

    print("\nAll outputs written to:", OUT_DIR)
    print("Files written:")
    for p in _written:
        print("  -", p)


if __name__ == "__main__":
    main()