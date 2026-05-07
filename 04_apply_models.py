"""
Apply the trained classifiers to the full post corpus.

The script loads the most recent timestamped model produced by
03_train_models.py, embeds every post in the corpus, and writes
three outputs: a record of every classified post, a long-format
trends CSV with monthly aggregations, and a top-K exemplar archive
per (month, stance, regime) cell for qualitative analysis.

Posts are read line by line from JSONL, embedded in
batches, classified, aggregated into running counters, and the
classified record is appended to disk after each batch.

"""

import os, heapq, itertools, glob
try:
    import ujson as json
except ImportError:
    import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from joblib import load
from sentence_transformers import SentenceTransformer
from config import (
    DATA_PATH, WORK_DIR, APPLY_BATCH_SIZE, TOP_K_EXEMPLARS,
    STANCE_MIN_CONF, REGIME_MIN_CONF, GIFT_MIN_CONF
)


def find_latest(pattern):
    """Return the most recently modified file matching a glob pattern, or None."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

STANCE_MODEL = find_latest(os.path.join(WORK_DIR, "stance_model_*.joblib")) \
               or os.path.join(WORK_DIR, "stance_model.joblib")
REGIME_MODEL = find_latest(os.path.join(WORK_DIR, "regime_model_*.joblib")) \
               or os.path.join(WORK_DIR, "regime_model.joblib")
GIFT_MODEL   = find_latest(os.path.join(WORK_DIR, "gift_model_*.joblib")) \
               or os.path.join(WORK_DIR, "gift_model.joblib")
ENCODERS     = os.path.join(WORK_DIR, "encoders.joblib")

OUT_TRENDS     = os.path.join(WORK_DIR, "monthly_trends_supervised.csv")
OUT_TOPJSONL   = os.path.join(WORK_DIR, "top_posts_by_month_stance_regime.jsonl")
OUT_CLASSIFIED = os.path.join(WORK_DIR, "classified_posts.jsonl")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Per-post helpers

ENG_FIELDS = ["likeCount", "repostCount", "replyCount", "quoteCount", "engagement", "eng"]

def engagement_weight(post):
 # sums every available engagement count field and adds +1 (posts with 0 engagement return 1)  
    total = 0.0
    found = False
    for f in ENG_FIELDS:
        v = post.get(f)
        if isinstance(v, (int, float)):
            total += float(v)
            found = True
    return (total + 1.0) if found else 1.0

def get_month(post):
    m = post.get("month_bucket")
    if isinstance(m, str) and len(m) >= 7:
        return m[:7]
    c = post.get("createdAt")
    if isinstance(c, str) and len(c) >= 7:
        return c[:7]
    return "unknown"

def get_id_fields(post):
    out = {}
    for k in ["uri", "cid", "id", "did", "handle"]:
        if k in post:
            out[k] = post[k]
    return out

def extract_full_text(post, for_model=True):
    parts = []

    main = post.get("text")
    if isinstance(main, str) and main.strip():
        parts.append(main.strip())

    image_alts = post.get("image_alts")
    if isinstance(image_alts, list):
        for alt in image_alts:
            if isinstance(alt, str) and alt.strip():
                prefix = "" if for_model else "[ALT] "
                parts.append(prefix + alt.strip())

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

def iter_batches(path, batch_size):
    batch = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main

def main():
 # applies the classifiers to the corpus and writes three output files

    for p in [STANCE_MODEL, REGIME_MODEL, ENCODERS]:
        if not p or not os.path.exists(p):
            raise FileNotFoundError(
                f"Required model file not found: {p}\n"
                "Run 03_train_models.py first."
            )

    print(f"Loading stance model:  {STANCE_MODEL}")
    print(f"Loading regime model:  {REGIME_MODEL}")

    stance_pack = load(STANCE_MODEL)
    regime_pack = load(REGIME_MODEL)
    enc         = load(ENCODERS)

    stance_clf       = stance_pack["model"]
    regime_clf       = regime_pack["model"]
    embed_model_name = stance_pack["embed_model"]

    regime_thresholds = regime_pack.get("thresholds", None)

    stance_le     = enc["stance_le"]
    mlb           = enc["mlb"]
    regime_labels = list(mlb.classes_)

    if regime_thresholds is None:
        print("WARNING: No tuned thresholds found in regime model. Using flat 0.5.")
        regime_thresholds = [0.5] * len(regime_labels)
    else:
        print("Loaded per-class regime thresholds from model.")

    gift_clf = None
    if GIFT_MODEL and os.path.exists(GIFT_MODEL):
        print(f"Loading gift model:    {GIFT_MODEL}")
        gift_pack = load(GIFT_MODEL)
        gift_clf  = gift_pack["model"]
    else:
        print("Gift model not found -- gift flag predictions will be skipped.")

    embedder = SentenceTransformer(embed_model_name)
    print(f"Embedding model: {embed_model_name}\n")

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Aggregators
    # two parallel aggregators for weighted sum and raw count

    # per-month totals.
    month_total_w = Counter()
    month_total_n = Counter()

    # per-month, per-stance and per-regime breakdowns.
    month_stance_w = defaultdict(Counter)
    month_stance_n = defaultdict(Counter)
    month_regime_w = defaultdict(Counter)
    month_regime_n = defaultdict(Counter)

    # per-month stance x regime cross-tabulation.
    month_stance_regime_w = defaultdict(lambda: defaultdict(Counter))
    month_stance_regime_n = defaultdict(lambda: defaultdict(Counter))

    # per-month gift-positive volume (weighted and raw)
    month_gift_w = Counter()
    month_gift_n = Counter()

    # top-K exemplars per (month, stance, regime)
    heaps     = defaultdict(list)
    _tiebreak = itertools.count()

    def push_topk(key, item, weight):
        # evicts the smallest item if the heap is full, otherwise just pushes
        entry = (weight, next(_tiebreak), item)
        h = heaps[key]
        if len(h) < TOP_K_EXEMPLARS:
            heapq.heappush(h, entry)
        else:
            if weight > h[0][0]:
                heapq.heapreplace(h, entry)

    total_raw       = 0
    total_processed = 0

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Main streaming loop

    with open(OUT_CLASSIFIED, "w", encoding="utf-8") as classified_out:

        for batch in tqdm(iter_batches(DATA_PATH, APPLY_BATCH_SIZE), desc="Applying models"):
            texts = []
            metas = []

            # phase 1: text construction. Posts without extractable text are skipped
            for post in batch:
                total_raw += 1
                text = extract_full_text(post, for_model=True)
                if not text:
                    continue

                # display text retains the [ALT] markers
                display_text = extract_full_text(post, for_model=False)

                month = get_month(post)
                w     = engagement_weight(post)

                texts.append(text)
                metas.append((post, month, w, display_text))
                month_total_w[month] += w
                month_total_n[month] += 1
                total_processed += 1

            if not texts:
                continue

            # phase 2: batch embedding and classification
            X = embedder.encode(
                texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
            )

            # stance: argmax across classes, retain max probability as the per-prediction confidence
            stance_proba = stance_clf.predict_proba(X)
            stance_conf  = stance_proba.max(axis=1)
            stance_pred  = stance_proba.argmax(axis=1)
            stance_lab   = stance_le.inverse_transform(stance_pred)

            # gift flag (retained, but not part of the thesis analysis)
            if gift_clf is not None:
                gift_proba = gift_clf.predict_proba(X)[:, 1]
                gift_pred  = (gift_proba >= 0.5).astype(int)
            else:
                gift_proba = np.zeros(len(texts))
                gift_pred  = np.zeros(len(texts), dtype=int)

            # regime: per-class probabilities
            reg_proba  = regime_clf.predict_proba(X)
            reg_binary = np.column_stack([
                (reg_proba[:, i] >= regime_thresholds[i]).astype(int)
                for i in range(len(regime_labels))
            ])

            # phase 3: per-post aggregation and output
            for (post, month, w, display_text), s, sconf, gprob, gpred, probs, rbin in zip(
                metas, stance_lab, stance_conf, gift_proba, gift_pred, reg_proba, reg_binary
            ):
                # stance confidence floor: route to the literal string uncertain rather than the argmax class
                if STANCE_MIN_CONF > 0 and sconf < STANCE_MIN_CONF:
                    s = "Uncertain"

                # gift confidence floor
                gift_uncertain = False
                if gift_clf is not None and GIFT_MIN_CONF > 0:
                    if not (gprob >= GIFT_MIN_CONF or gprob <= (1.0 - GIFT_MIN_CONF)):
                        gift_uncertain = True

                regimes = [regime_labels[i] for i, v in enumerate(rbin) if v == 1]

                # optional regime confidence floor: discard regimes that fired above their tuned threshold, but below the global REGIME_MIN_CONF 
                if REGIME_MIN_CONF > 0 and regimes:
                    regimes = [r for r in regimes
                               if probs[regime_labels.index(r)] >= REGIME_MIN_CONF]

                # posts with no regime above any threshold get the Other_Unclear label
                if not regimes:
                    regimes = ["Other_Unclear"]

                # per-regime probabilities are stored for analysing alternative thresholds without re-running the embedder and classifier
                regime_confs = {
                    regime_labels[i]: round(float(probs[i]), 4)
                    for i in range(len(regime_labels))
                }

                # update stance and regime aggregators (weighted and raw counts)
                month_stance_w[month][s] += w
                month_stance_n[month][s] += 1
                for r in regimes:
                    month_regime_w[month][r] += w
                    month_regime_n[month][r] += 1
                    month_stance_regime_w[month][s][r] += w
                    month_stance_regime_n[month][s][r] += 1

                # update gift volume aggregators only
                if gift_clf is not None and not gift_uncertain and int(gpred) == 1:
                    month_gift_w[month] += w
                    month_gift_n[month] += 1

                # write one record to classified_posts.jsonl
                classified_record = {
                    "uri":           post.get("uri", ""),
                    "author_did":    post.get("author_did", post.get("did", "")),
                    "author_handle": post.get("author_handle", post.get("handle", "")),
                    "month":         month,
                    "text":          display_text,
                    "stance":        s,
                    "stance_conf":   round(float(sconf), 4),
                    "regimes":       "|".join(regimes),
                    "regime_confs":  regime_confs,
                    "gift_flag":     int(gpred),
                    "gift_conf":     round(float(gprob), 4),
                    "weight":        round(float(w), 4),
                }
                classified_out.write(
                    json.dumps(classified_record, ensure_ascii=False) + "\n"
                )

                # push to the per-cell exemplar heaps. A post that carries multiple regimes is pushed once per regime, same exemplar can appear in multiple cells
                for r in regimes:
                    item = {
                        "month":       month,
                        "stance":      s,
                        "regime":      r,
                        "weight":      float(w),
                        "stance_conf": float(sconf),
                        "gift_flag":   int(gpred),
                        "gift_conf":   float(gprob),
                        "text":        display_text,
                    }
                    item.update(get_id_fields(post))
                    push_topk((month, s, r), item, w)

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Write trends CSV

    rows = []
    months = sorted(month_total_w.keys())

    all_stances = sorted({s for m in month_stance_w for s in month_stance_w[m]})
    all_regimes = sorted({r for m in month_regime_w for r in month_regime_w[m]})

    for m in months:
        tot = float(month_total_w[m]) if month_total_w[m] else 0.0
        tot_n = float(month_total_n[m]) if month_total_n[m] else 0.0

        # stance rows
        for s in all_stances:
            wsum = float(month_stance_w[m].get(s, 0.0))
            nsum = float(month_stance_n[m].get(s, 0.0))
            rows.append({"month": m, "kind": "stance", "label": s,
                         "weight_sum": wsum, "share": (wsum / tot if tot else 0.0),
                         "post_count": nsum, "count_share": (nsum / tot_n if tot_n else 0.0)})

        # regime rows
        for r in all_regimes:
            wsum = float(month_regime_w[m].get(r, 0.0))
            nsum = float(month_regime_n[m].get(r, 0.0))
            rows.append({"month": m, "kind": "regime", "label": r,
                         "weight_sum": wsum, "share": (wsum / tot if tot else 0.0),
                         "post_count": nsum, "count_share": (nsum / tot_n if tot_n else 0.0)})

        # stance x regime cross-tabulation rows
        for s in all_stances:
            for r in all_regimes:
                wsum = float(month_stance_regime_w[m][s].get(r, 0.0))
                nsum = float(month_stance_regime_n[m][s].get(r, 0.0))
                if wsum > 0 or nsum > 0:
                    rows.append({"month": m, "kind": "stance_x_regime",
                                 "label": f"{s}__{r}",
                                 "weight_sum": wsum,
                                 "share": (wsum / tot if tot else 0.0),
                                 "post_count": nsum,
                                 "count_share": (nsum / tot_n if tot_n else 0.0)})

        # gift volume row
        gw  = float(month_gift_w.get(m, 0.0))
        gn  = float(month_gift_n.get(m, 0.0))
        rows.append({"month": m, "kind": "gift_flag", "label": "gift_reciprocity",
                     "weight_sum": gw, "share": (gw / tot if tot else 0.0),
                     "post_count": gn, "count_share": (gn / tot_n if tot_n else 0.0)})

    pd.DataFrame(rows).to_csv(OUT_TRENDS, index=False)

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Write top exemplars JSONL
    with open(OUT_TOPJSONL, "w", encoding="utf-8") as f:
        for (month, stance, regime), h in sorted(heaps.items()):
            for w, _, item in sorted(h, key=lambda x: x[0], reverse=True):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nDONE")
    print(f"Raw lines read from JSONL:          {total_raw:,}")
    print(f"Posts with extractable text:        {total_processed:,}")
    print(f"Posts skipped (no text):            {total_raw - total_processed:,}")
    print(f"Wrote trends CSV:                   {OUT_TRENDS}")
    print(f"Wrote exemplar archive:             {OUT_TOPJSONL}")
    print(f"Wrote classified posts:             {OUT_CLASSIFIED}")

if __name__ == "__main__":
    main()