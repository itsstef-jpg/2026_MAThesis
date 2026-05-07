# 07_collocation_concordance.py
"""
Collocation, n-gram, and concordance analysis for the corpus.

"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from config import WORK_DIR

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Configuration

KEYNESS_CSV = os.path.join(WORK_DIR, "keyness", "keyness_all_regimes.csv")
TOP_N_TERMS_PER_REGIME = 10

TARGETS_OVERRIDE = None

CLASSIFIED_PATH = os.path.join(WORK_DIR, "classified_posts.jsonl")

# output directory
OUT_DIR = os.path.join(WORK_DIR, "collocation")
os.makedirs(OUT_DIR, exist_ok=True)

# window size for collocation. AntConc default is 5L/5R. Bluesky posts are short so 4L/4R
WINDOW = 4

# how many of each output to keep
TOP_COLLOCATES = 30
TOP_NGRAMS     = 25
N_CONCORDANCE  = 20    # how many concordance lines per (regime, term) per sample type
CONC_WIDTH     = 60    # characters of context to either side of focus term

# minimum collocate frequency to consider, filters out single-occurrence noise
MIN_COLLOCATE_FREQ = 5

random.seed(42)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Target derivation
# reads the keyness CSV and selects the top-N most distinctive terms per regime


def derive_targets_from_keyness(keyness_path: str, top_n: int) -> List[Tuple[str, str]]:

    if not os.path.exists(keyness_path):
        print(f"ERROR: keyness CSV not found at {keyness_path}")
        print("Either run the keyness analysis first or set TARGETS_OVERRIDE.")
        return []

    k = pd.read_csv(keyness_path)
    targets: List[Tuple[str, str]] = []
    for regime, group in k.groupby("regime"):
        top = group.sort_values("log_likelihood", ascending=False).head(top_n)
        for _, row in top.iterrows():
            term = str(row["term"]).strip().lower()
            if not term:
                continue
            targets.append((regime, term))
    return targets

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Tokenisation
# same approach as the keyness pipeline in script 06

TOKEN_RE = re.compile(r"[a-z][a-z'-]*")

def tokenize(text: str) -> List[str]:

    if not text:
        return []
    tokens = TOKEN_RE.findall(text.lower())
    result = []
    for tok in tokens:
        if "'" in tok:
            stem = tok.split("'")[0]
            if stem in STOPWORDS:
                continue  # drop: don't, isn't, won't, etc.
        result.append(tok)
    return result


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Stopword list

STOPWORDS = {
    # articles
    "the", "a", "an",
    # coordinating conjunctions
    "and", "or", "but", "so",
    # prepositions
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "into",
    "up", "out", "over", "after", "about",
    # auxiliary verbs and copulas
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "will", "would", "could", "should",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    # filler
    "all", "any", "some", "more", "just", "like", "than", "then",
    "also", "yes",
    # tokenisation artefacts: contraction fragments after splitting on apostrophes
    "s", "t", "re", "ve", "ll", "d", "m",
    # ALT-text marker fragment that survives punctuation stripping
    "alt",
    "don", "isn", "won", "didn", "doesn", "wouldn", "couldn", "shouldn",
    "haven", "hadn", "wasn", "weren", "aren",
}


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Streaming reader

def stream_posts(path: str):

    shown = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                p = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue

            # Field-priority for regimes — multi-label or single-label
            regimes = p.get("regimes") or p.get("predicted_regimes")
            if regimes is None:
                single = p.get("regime")
                regimes = [single] if single else []
            # Handle pipe-delimited string format (some inference outputs store
            # multi-label assignments as 'A|B|C' strings rather than as lists).
            if isinstance(regimes, str):
                regimes = [r.strip() for r in regimes.split("|") if r.strip()]

            if not shown and regimes:
                print(f"DEBUG first post regimes field: {regimes!r} (type={type(regimes).__name__})")
                shown = True

            text = p.get("text", "") or ""
            weight = p.get("weight") or p.get("engagement") or 0
            uri = p.get("uri", "")
            yield (regimes, text, weight, uri)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# pass 1: Collect target-relevant data while streaming
# ----------------------------------------------------------------------
# A single streaming pass through the classified-posts JSONL
# accumulates everything needed for the per-target outputs:
#
# - window_tokens / window_total: token counts inside the WINDOW- wide window around each occurrence of the focus term, used as the target half of the log-likelihood collocation.
# - regime_tokens / regime_total: token counts across the whole regime, used as the reference half. The reference is the regime as a whole rather than the rest of the corpus, so the LL ranks terms that are distinctive of the focus term within this regime's discourse rather than terms that are distinctive of the regime overall.
# - post_records: list of (text, weight, uri) tuples for posts that contain the focus term, used to build concordances in pass 2.
# - ngram_bigrams / ngram_trigrams: bigram and trigram counts across the post (not the window) for posts containing the focus term.

def collect_target_data(path: str, targets: List[Tuple[str, str]]):
    target_set = set(targets)
    target_regimes = {r for (r, _) in targets}

    state = {tgt: {
        "window_tokens": Counter(),
        "window_total":  0,
        "regime_tokens": Counter(),
        "regime_total":  0,
        "post_records":  [],
        "ngram_bigrams": Counter(),
        "ngram_trigrams": Counter(),
    } for tgt in targets}

    n_posts = 0
    n_with_text = 0

    for regimes, text, weight, uri in stream_posts(path):
        n_posts += 1
        if not text:
            continue
        n_with_text += 1

        # skip if this post doesn't belong to any target regime
        regimes_in_targets = [r for r in regimes if r in target_regimes]
        if not regimes_in_targets:
            continue

        toks = tokenize(text)
        if not toks:
            continue

        for regime in regimes_in_targets:
            # update regime-wide token counts for every (regime, *) target
            for tgt in targets:
                if tgt[0] != regime:
                    continue
                st = state[tgt]
                st["regime_tokens"].update(toks)
                st["regime_total"] += len(toks)

                term = tgt[1]
                # find positions of the focus term
                positions = [i for i, t in enumerate(toks) if t == term]
                if not positions:
                    continue

                # record this post for concordance sampling
                st["post_records"].append((text, weight, uri))

                # walk each occurrence and accumulate window tokens
                for pos in positions:
                    start = max(0, pos - WINDOW)
                    end = min(len(toks), pos + WINDOW + 1)
                    for j in range(start, end):
                        if j == pos:
                            continue
                        st["window_tokens"][toks[j]] += 1
                        st["window_total"] += 1

                # n-grams: bigrams and trigrams within the post (not the window)
                # only counted from posts that contain the focus term
                for i in range(len(toks) - 1):
                    bg = (toks[i], toks[i+1])
                    if bg[0] in STOPWORDS and bg[1] in STOPWORDS:
                        continue
                    st["ngram_bigrams"][bg] += 1
                for i in range(len(toks) - 2):
                    tg = (toks[i], toks[i+1], toks[i+2])
                    # drop trigrams that are all stopwords
                    if all(t in STOPWORDS for t in tg):
                        continue
                    st["ngram_trigrams"][tg] += 1

        if n_posts % 100000 == 0:
            print(f"  ...processed {n_posts:,} posts")

    print(f"\nTotal posts read: {n_posts:,}")
    print(f"Posts with text:  {n_with_text:,}")
    return state


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Log-likelihood

def loglikelihood(o11: int, o12: int, o21: int, o22: int) -> float:

    n = o11 + o12 + o21 + o22
    if n == 0:
        return 0.0

    # expected counts
    e11 = ((o11 + o12) * (o11 + o21)) / n
    e12 = ((o11 + o12) * (o12 + o22)) / n
    e21 = ((o21 + o22) * (o11 + o21)) / n
    e22 = ((o21 + o22) * (o12 + o22)) / n

    def cell(o, e):
        if o == 0 or e == 0:
            return 0.0
        return o * math.log(o / e)

    g2 = 2 * (cell(o11, e11) + cell(o12, e12) + cell(o21, e21) + cell(o22, e22))
    # sign by direction of association
    if e11 > 0 and o11 < e11:
        g2 = -g2
    return g2


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Pass 2: Compute collocation LL, n-grams, and concordances

def compute_collocates(st: Dict) -> pd.DataFrame:

    rows = []
    window_total = st["window_total"]
    regime_total = st["regime_total"]
    if window_total == 0 or regime_total == 0:
        return pd.DataFrame(columns=["collocate", "window_count", "regime_count",
                                     "window_per_10k", "regime_per_10k", "log_likelihood"])

    out_of_window_total = regime_total - window_total
    for term, win_count in st["window_tokens"].items():
        if win_count < MIN_COLLOCATE_FREQ:
            continue
        if term in STOPWORDS:
            continue
        regime_count_term = st["regime_tokens"].get(term, 0)
        out_of_window_count = regime_count_term - win_count
        if out_of_window_count < 0:
            out_of_window_count = 0  # shouldn't happen but defensive

        ll = loglikelihood(
            o11=win_count,
            o12=window_total - win_count,
            o21=out_of_window_count,
            o22=out_of_window_total - out_of_window_count,
        )
        if ll <= 0:
            continue  # skip underrepresented terms

        rows.append({
            "collocate":      term,
            "window_count":   win_count,
            "regime_count":   regime_count_term,
            "window_per_10k": (win_count / window_total) * 10000,
            "regime_per_10k": (regime_count_term / regime_total) * 10000,
            "log_likelihood": ll,
        })

    df = pd.DataFrame(rows).sort_values("log_likelihood", ascending=False).reset_index(drop=True)
    return df


def compute_ngrams(st: Dict, n_top: int = TOP_NGRAMS) -> Tuple[pd.DataFrame, pd.DataFrame]:
 # return top-n bigrams and trigrams as dataframes   
    bg_rows = [
        {"ngram": " ".join(bg), "count": c}
        for bg, c in st["ngram_bigrams"].most_common(n_top)
    ]
    tg_rows = [
        {"ngram": " ".join(tg), "count": c}
        for tg, c in st["ngram_trigrams"].most_common(n_top)
    ]
    return pd.DataFrame(bg_rows), pd.DataFrame(tg_rows)


def compute_concordances(st: Dict, term: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    records = st["post_records"]
    if not records:
        empty = pd.DataFrame(columns=["left", "focus", "right", "weight", "uri"])
        return empty, empty

    pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)

    def kwic_lines(record):
        text, weight, uri = record
        lines = []
        for m in pat.finditer(text):
            l = max(0, m.start() - CONC_WIDTH)
            r = min(len(text), m.end() + CONC_WIDTH)
            left = text[l:m.start()].replace("\n", " ").strip() or "(start)"
            focus = text[m.start():m.end()]
            right = text[m.end():r].replace("\n", " ").strip() or "(end)"
            lines.append({"left": left, "focus": focus, "right": right,
                          "weight": weight, "uri": uri})
        return lines

    # build full pool of concordance lines (one per occurrence)
    pool = []
    for rec in records:
        pool.extend(kwic_lines(rec))

    if not pool:
        empty = pd.DataFrame(columns=["left", "focus", "right", "weight", "uri"])
        return empty, empty

    # random sample
    sample_n = min(N_CONCORDANCE, len(pool))
    random_sample = random.sample(pool, sample_n)

    # Top by engagement
    top_by_weight = sorted(pool, key=lambda r: -float(r.get("weight", 0)))[:N_CONCORDANCE]

    return pd.DataFrame(random_sample), pd.DataFrame(top_by_weight)


def safe_filename(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main

def main():
 # run target derivation, the streaming pass, and per-target output writing
    print(f"Streaming classified posts from: {CLASSIFIED_PATH}")
    if not os.path.exists(CLASSIFIED_PATH):
        print(f"ERROR: classified posts file not found at {CLASSIFIED_PATH}")
        print("Adjust CLASSIFIED_PATH at the top of the script.")
        return

    # build target list - either auto-derived from keyness data or from override
    if TARGETS_OVERRIDE is not None:
        targets = list(TARGETS_OVERRIDE)
        print(f"Using TARGETS_OVERRIDE ({len(targets)} pairs)")
    else:
        targets = derive_targets_from_keyness(KEYNESS_CSV, TOP_N_TERMS_PER_REGIME)
        if not targets:
            print("No targets derived; cannot proceed.")
            return
        print(f"Derived {len(targets)} targets from {KEYNESS_CSV}")
        print(f"  (top {TOP_N_TERMS_PER_REGIME} terms per regime by log-likelihood)")

    # show first few for verification
    print("\nFirst 5 targets:")
    for tgt in targets[:5]:
        print(f"  {tgt}")
    if len(targets) > 5:
        print(f"  ... and {len(targets) - 5} more")
    print()

    state = collect_target_data(CLASSIFIED_PATH, targets)

    written = []

    # organise output into per-regime subdirectories for navigability at scale
    for tgt in targets:
        regime, term = tgt
        st = state[tgt]
        n_records = len(st["post_records"])
        regime_total = st["regime_total"]
        if n_records == 0:
            print(f"  SKIP {regime} / {term!r}: posts containing term = 0, regime tokens = {regime_total:,}")
        else:
            print(f"  WRITE {regime} / {term!r}: posts = {n_records}, regime tokens = {regime_total:,}")
        slug = safe_filename(term)
        regime_dir = os.path.join(OUT_DIR, safe_filename(regime))
        os.makedirs(regime_dir, exist_ok=True)

        n_records = len(st["post_records"])
        if n_records == 0:
            # no posts in this regime contained the term - skip silently
            continue

        # Collocates
        coloc_df = compute_collocates(st)
        if not coloc_df.empty:
            top_coloc = coloc_df.head(TOP_COLLOCATES)
            path = os.path.join(regime_dir, f"collocates_{slug}.csv")
            top_coloc.to_csv(path, index=False, float_format="%.4f")
            written.append(path)

        # N-grams
        bg_df, tg_df = compute_ngrams(st)
        if not bg_df.empty:
            path = os.path.join(regime_dir, f"bigrams_{slug}.csv")
            bg_df.to_csv(path, index=False)
            written.append(path)
        if not tg_df.empty:
            path = os.path.join(regime_dir, f"trigrams_{slug}.csv")
            tg_df.to_csv(path, index=False)
            written.append(path)

        # Concordances
        rand_df, top_df = compute_concordances(st, term)
        if not rand_df.empty:
            path = os.path.join(regime_dir, f"concordance_{slug}_random.csv")
            rand_df.to_csv(path, index=False)
            written.append(path)
        if not top_df.empty:
            path = os.path.join(regime_dir, f"concordance_{slug}_top_engagement.csv")
            top_df.to_csv(path, index=False)
            written.append(path)

    # Cross-regime comparison files

    compare_dir = os.path.join(OUT_DIR, "_cross_regime_comparisons")
    os.makedirs(compare_dir, exist_ok=True)

    term_to_regimes: Dict[str, List[str]] = defaultdict(list)
    for regime, term in targets:
        term_to_regimes[term].append(regime)

    n_comparisons_written = 0
    for term, regime_list in term_to_regimes.items():
        if len(regime_list) < 2:
            continue

        per_regime_top: Dict[str, pd.DataFrame] = {}
        for regime in regime_list:
            tgt = (regime, term)
            st = state[tgt]
            df = compute_collocates(st)
            if df.empty:
                continue
            per_regime_top[regime] = df.head(TOP_COLLOCATES).set_index("collocate")[
                ["log_likelihood", "window_count"]
            ].rename(columns={
                "log_likelihood": f"LL__{regime}",
                "window_count":   f"count__{regime}",
            })

        if len(per_regime_top) < 2:
            continue

        # Outer-join all regime tables on collocate
        combined = None
        for df in per_regime_top.values():
            combined = df if combined is None else combined.join(df, how="outer")
        combined = combined.fillna(0).reset_index().rename(columns={"index": "collocate"})

        # sort by sum of LLs across regimes
        ll_cols = [c for c in combined.columns if c.startswith("LL__")]
        combined["_sum_ll"] = combined[ll_cols].sum(axis=1)
        combined = combined.sort_values("_sum_ll", ascending=False).drop(columns=["_sum_ll"])

        path = os.path.join(compare_dir, f"compare_{safe_filename(term)}.csv")
        combined.to_csv(path, index=False, float_format="%.4f")
        written.append(path)
        n_comparisons_written += 1

    if n_comparisons_written:
        print(f"\nWrote {n_comparisons_written} cross-regime comparison files to {compare_dir}")

    # Summary markdown
    summary_md = os.path.join(OUT_DIR, "collocation_summary.md")
    with open(summary_md, "w") as f:
        f.write("# Collocation, n-gram, and concordance outputs\n\n")
        f.write(f"- Window size: {WINDOW}L/{WINDOW}R\n")
        f.write(f"- Min collocate frequency: {MIN_COLLOCATE_FREQ}\n")
        f.write(f"- Concordance lines per sample: {N_CONCORDANCE}\n")
        f.write(f"- Top collocates retained: {TOP_COLLOCATES}\n")
        f.write(f"- Total targets analysed: {len(targets)}\n")
        f.write(f"- Cross-regime comparison files: {n_comparisons_written}\n\n")

        # Per-regime summary
        regime_to_targets: Dict[str, List[str]] = defaultdict(list)
        for r, t in targets:
            regime_to_targets[r].append(t)

        f.write("## Targets analysed per regime\n\n")
        for regime in sorted(regime_to_targets):
            terms = regime_to_targets[regime]
            f.write(f"### {regime}\n\n")
            f.write(f"Subdirectory: `{safe_filename(regime)}/`\n\n")
            f.write("Terms analysed: ")
            f.write(", ".join(f"`{t}`" for t in terms))
            f.write("\n\n")
            # Note posts containing each term
            for term in terms:
                tgt = (regime, term)
                if tgt in state:
                    n = len(state[tgt]["post_records"])
                    if n == 0:
                        f.write(f"- `{term}`: 0 posts (no output files)\n")
                    else:
                        f.write(f"- `{term}`: {n:,} posts\n")
            f.write("\n")

        if n_comparisons_written:
            f.write("## Cross-regime comparisons\n\n")
            f.write(f"Subdirectory: `_cross_regime_comparisons/`\n\n")
            f.write("Terms appearing in multiple regimes' top lists are compared "
                    "side-by-side, with collocates from each regime shown in their own "
                    "log-likelihood column.\n\n")
            for term, regime_list in term_to_regimes.items():
                if len(regime_list) >= 2:
                    f.write(f"- `{term}`: in {len(regime_list)} regimes "
                            f"({', '.join(regime_list)})\n")

    written.append(summary_md)

    print(f"\nWrote {len(written)} files total to {OUT_DIR}")
    print(f"  Summary index: {summary_md}")


if __name__ == "__main__":
    main()