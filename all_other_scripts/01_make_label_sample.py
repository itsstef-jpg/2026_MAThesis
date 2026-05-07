
"""
Stratified, engagement-weighted sampling for manual labelling.

This script reads the consolidated post corpus (a JSONL file produced
by the fetcher) and writes a labelling spreadsheet (CSV) containing a
fixed-size random sample drawn under two constraints:

  1. Stratification by calendar month. Each month present in the corpus
     receives an allocation proportional to its share of total posts,
     with a per-month minimum so that low-volume months are still
     visible in the labelled set.
  2. Engagement weighting. A configurable fraction
     of each month's allocation is drawn from the highest-engagement
     posts in that month (`TOP_ENGAGEMENT_FRAC`), the remainder is
     drawn uniformly at random from the rest of the month's pool.

The output CSV is the input to the Streamlit labelling app.

"""

import os, random
try:
    import ujson as json
except ImportError:
    import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from config import DATA_PATH, WORK_DIR, LABEL_SAMPLE_N, TOP_ENGAGEMENT_FRAC, RANDOM_SEED

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Engagement weighting
# sum of all engagement-count fields plus one, so posts with zero engagement still have a non-zero weight and can appear in the random remainder

ENG_FIELDS = ["likeCount", "repostCount", "replyCount", "quoteCount", "engagement", "eng"]

def engagement_weight(post):
    total = 0.0
    found = False
    for f in ENG_FIELDS:
        v = post.get(f)
        if isinstance(v, (int, float)):
            total += float(v)
            found = True
    return (total + 1.0) if found else 1.0

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Month and ID extraction

def get_month(post):
 # return the calendar month of a post as a YYYY-MM string  
    m = post.get("month_bucket")
    if isinstance(m, str) and len(m) >= 7:
        return m[:7]
    c = post.get("createdAt")
    if isinstance(c, str) and len(c) >= 7:
        return c[:7]
    return "unknown"

def get_id(post, idx):
 # return identifier for a post   
    return post.get("uri") or post.get("cid") or post.get("id") or f"row_{idx}"

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Text construction

def extract_full_text(post, for_model=False):
 # combine post text + image ALT text if present (with [ALT] markers for provenance) 
    parts = []

    # main post text
    main = post.get("text")
    if isinstance(main, str) and main.strip():
        parts.append(main.strip())

    # primary alt text source
    image_alts = post.get("image_alts")
    if isinstance(image_alts, list):
        for alt in image_alts:
            if isinstance(alt, str) and alt.strip():
                prefix = "" if for_model else "[ALT] "
                parts.append(prefix + alt.strip())

    # secondary alt text source, nested embed.images path
    embed = post.get("embed")
    if isinstance(embed, dict):
        images = embed.get("images")
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    alt = img.get("alt")
                    if isinstance(alt, str) and alt.strip():
                        prefix = "" if for_model else "[ALT] "
                        parts.append(prefix + alt.strip())

    return " ".join(parts).replace("\n", " ").strip()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main

def main():
 # build and write the stratified labelling sample
    os.makedirs(WORK_DIR, exist_ok=True)
    out_csv = os.path.join(WORK_DIR, "labels_sample.csv")
    random.seed(RANDOM_SEED)

    # pass 1: month counts
    month_counts = defaultdict(int)
    total = 0
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting months"):
            if not line.strip():
                continue
            post = json.loads(line)
            month_counts[get_month(post)] += 1
            total += 1

    # separate real months from unknown
    unknown_count = month_counts.pop("unknown", 0)
    if unknown_count:
        print(f"Note: {unknown_count} post(s) had an unparseable month and were excluded from sampling.")
    real_total = total - unknown_count

    months = sorted(month_counts.keys())

    # allocation
    min_per_month = max(10, LABEL_SAMPLE_N // (len(months) * 4))

    alloc = {}
    remaining = LABEL_SAMPLE_N
    for m in months:
        alloc[m] = max(min_per_month, int(LABEL_SAMPLE_N * (month_counts[m] / real_total)))
        remaining -= alloc[m]

    # reconcile rounding
    if remaining > 0:
        # distribute leftovers one at a time across months in fixed order
        while remaining > 0:
            for m in months:
                alloc[m] += 1
                remaining -= 1
                if remaining <= 0:
                    break
    elif remaining < 0:
        for m in sorted(months, key=lambda x: alloc[x], reverse=True):
            while alloc[m] > min_per_month and remaining < 0:
                alloc[m] -= 1
                remaining += 1
            if remaining == 0:
                break

    # pass 2: collect per-month candidates
    buckets = defaultdict(list)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Collecting candidates")):
            if not line.strip():
                continue
            post = json.loads(line)

            month = get_month(post)
            # skip posts without a real month
            if month == "unknown":
                continue

            text = extract_full_text(post, for_model=False)
            if not text:
                # drop posts with no usable text
                continue

            w = engagement_weight(post)

            buckets[month].append({
                "sample_id": get_id(post, idx),
                "month": month,
                "weight": w,
                "text": text,
            })

    # per-month draw, cross-month deduplication, and final shuffle
    final = []
    for m in months:
        n = alloc[m]
        n_top = int(n * TOP_ENGAGEMENT_FRAC)
        n_rand = n - n_top

        pool = buckets[m]
        if not pool:
            continue

        pool_sorted = sorted(pool, key=lambda x: x["weight"], reverse=True)

        # top-engagement slice
        final.extend(pool_sorted[:n_top])

        # random remainder, drawn from the posts not already taken
        remainder = pool_sorted[n_top:]
        if remainder:
            final.extend(random.sample(remainder, k=min(n_rand, len(remainder))))

    # cross-month deduplication
    seen = set()
    dedup = []
    for r in final:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        dedup.append(r)

    # shuffle month order
    df = pd.DataFrame(dedup).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df["stance"] = ""
    df["regimes"] = ""
    df["gift_reciprocity_flag"] = ""
    df["notes"] = ""

    # every field wrapped in quotes to prevent being misread as cell boundaries
    df.to_csv(out_csv, index=False, quoting=1)
    print("Wrote:", out_csv)
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
