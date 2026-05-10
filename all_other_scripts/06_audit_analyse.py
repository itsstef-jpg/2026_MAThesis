
"""
Audit, keyness, temporal-term, and prolific-author analyses.

This script uses classified_posts.jsonl

Multi-label note for keyness: A post carrying two regime labels has its tokens added to BOTH regimes' counts. This reflects the multi-label design of the taxonomy: a post participating in two narrative logics simultaneously contributes its vocabulary to both. The consequence is that terms shared between co-occurring regimes will appear prominent in both keyness tables.

"""

import os
import re
import math
import random
try:
    import ujson as json
except ImportError:
    import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import (
    WORK_DIR, REGIME_LABELS, STANCE_LABELS, RANDOM_SEED
)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Paths
CLASSIFIED_JSONL = os.path.join(WORK_DIR, "classified_posts.jsonl")
TOPJSONL         = os.path.join(WORK_DIR, "top_posts_by_month_stance_regime.jsonl")

AUDIT_DIR    = os.path.join(WORK_DIR, "audit")
KEYNESS_DIR  = os.path.join(WORK_DIR, "keyness")
TEMPORAL_DIR = os.path.join(WORK_DIR, "temporal_terms")
ROBUST_DIR   = os.path.join(WORK_DIR, "robustness")

for d in [AUDIT_DIR, KEYNESS_DIR, TEMPORAL_DIR, ROBUST_DIR]:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Settings
AUDIT_SAMPLE_N  = 50  # posts per regime for audit
PROLIFIC_TOP_N  = 20  # number of top authors to remove for robustness check
KEYNESS_TOP_N   = 30  # top N distinctive terms to report per regime
TEMPORAL_TOP_N  = 15  # top N distinctive terms per regime per month
MIN_TERM_FREQ   = 5   # ignore terms appearing fewer than this many times across the full classified corpus (filters noise/typos)

random.seed(RANDOM_SEED)
_written = []

def wrote(path):
    _written.append(path)
    print("Wrote:", path)

# ----------------------------------------------------------------------
# Text cleaning and tokenisation
# ----------------------------------------------------------------------
# a simple regex and stopword tokeniser. deliberately retains subordinating conjunctions and negations because in this corpus they are often content words (e.g. because, while, if, not, no). also explicitly filters tokenisation artefacts that survive punctuation stripping

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
# Display settings

REGIME_DISPLAY_NAMES = {
    "Extraction_Dispossession":                "Extraction & Dispossession",
    "Human_Essence_Ontology":                  "Human Essence & Ontology",
    "Aesthetic_Pollution_Epistemic_Corruption": "Aesthetic Pollution",
    "Governance_Boundary_Policing":            "Governance & Boundary Policing",
    "Ideology_Hype_Discourse_Wars":            "Ideology, Hype & Discourse Wars",
    "AI_Native_Subculture_Legitimation":       "AI-Native Subculture",
    "Human_Artist_Community_Reproduction":     "Human Artist Community",
    "Adult_Content_NSFWAIGen":                 "Adult Content & NSFW AI",
    "Other_Unclear":                           "Other / Unclear",
}

LABEL_COLORS = {
    "Extraction_Dispossession":                "#e63946",
    "Human_Essence_Ontology":                  "#457b9d",
    "Aesthetic_Pollution_Epistemic_Corruption": "#f4a261",
    "Governance_Boundary_Policing":            "#2a9d8f",
    "Ideology_Hype_Discourse_Wars":            "#9b2226",
    "AI_Native_Subculture_Legitimation":       "#6a4c93",
    "Human_Artist_Community_Reproduction":     "#40916c",
    "Adult_Content_NSFWAIGen":                 "#e9c46a",
    "Other_Unclear":                           "#adb5bd",
    "pro":     "#2d6a4f",
    "anti":    "#c1121f",
    "mixed":   "#f4a261",
    "neutral": "#adb5bd",
}

def display_name(label):
    return REGIME_DISPLAY_NAMES.get(label, label)

def label_color(label):
    return LABEL_COLORS.get(label, None)

PLOT_REGIME_FILTERS = {
}

def apply_regime_filter(columns, plot_key):
    f = PLOT_REGIME_FILTERS.get(plot_key)
    if f is None:
        return columns
    if "include" in f:
        keep = [c for c in columns if c in f["include"]]
        return keep if keep else columns
    if "exclude" in f:
        keep = [c for c in columns if c not in f["exclude"]]
        return keep if keep else columns
    return columns

def tokenize(text):
    text = text.lower()
    text = re.sub(r"\[alt\]\s*", " ", text)   # remove [ALT] markers explicitly
    text = re.sub(r"[^\w\s]", " ", text)       # punctuation -> space
    text = re.sub(r"\d+", " ", text)           # numbers -> space
    tokens = text.split()
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Log-likelihood keyness
# 2 * sum(0 * ln() / E)) (same metric used by AntConc and other corpus tools)


def log_likelihood(freq_a, total_a, freq_b, total_b):
    """
    Compute log-likelihood keyness (G2) for a single term.

    freq_a  : frequency of term in target corpus (a regime)
    total_a : total token count in target corpus
    freq_b  : frequency of term in reference corpus (rest of corpus)
    total_b : total token count in reference corpus

    Higher G2 = more distinctive. Positive = overrepresented in target.
    """
    e_a = total_a * (freq_a + freq_b) / (total_a + total_b)
    e_b = total_b * (freq_a + freq_b) / (total_a + total_b)

    def safe_log(o, e):
        if o == 0 or e == 0:
            return 0.0
        return o * math.log(o / e)

    g2 = 2 * (safe_log(freq_a, e_a) + safe_log(freq_b, e_b))
    # preserve sign: negative means underrepresented in target
    if freq_a / (total_a + 1e-9) < freq_b / (total_b + 1e-9):
        g2 = -g2
    return g2

def compute_keyness(regime_counts, regime_total, corpus_counts, corpus_total, top_n):
 # return a dataframe of the top_n most distinctive terms for a regime  
 
    rows = []
    for term, freq_a in regime_counts.items():
        if freq_a < MIN_TERM_FREQ:
            continue
        freq_b  = corpus_counts.get(term, 0) - freq_a
        total_b = corpus_total - regime_total
        if total_b <= 0:
            continue
        g2 = log_likelihood(freq_a, regime_total, freq_b, total_b)
        rows.append({
            "term":           term,
            "regime_freq":    freq_a,
            "corpus_freq":    corpus_counts.get(term, 0),
            "regime_per_10k": round(freq_a / regime_total * 10000, 2) if regime_total else 0,
            "corpus_per_10k": round(corpus_counts.get(term, 0) / corpus_total * 10000, 2) if corpus_total else 0,
            "log_likelihood": round(g2, 3),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("log_likelihood", ascending=False).head(top_n).reset_index(drop=True)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Streaming pass through classified_posts.jsonl

def streaming_pass_classified():

    print("\n" + "="*60)
    print("STREAMING PASS through classified_posts.jsonl")
    print("="*60)

    if not os.path.exists(CLASSIFIED_JSONL):
        print(f"ERROR: classified_posts.jsonl not found at {CLASSIFIED_JSONL}")
        print("Run 04_apply_models_streaming.py first.")
        return None

    # part 1: audit buckets — regime -> confidence band -> list of posts
    audit_buckets = defaultdict(lambda: {"high": [], "mid": [], "low": []})

    # parts 2-3: term counts
    corpus_term_counts  = Counter()
    corpus_total_tokens = 0

    regime_term_counts  = defaultdict(Counter)
    regime_total_tokens = defaultdict(int)

    regime_month_term_counts  = defaultdict(lambda: defaultdict(Counter))
    regime_month_total_tokens = defaultdict(lambda: defaultdict(int))

    # part 4: author counts
    author_regime_counts = defaultdict(Counter)
    author_stance_counts = defaultdict(Counter)
    author_total_counts  = Counter()
    author_handles       = {}

    total_posts = 0

    with open(CLASSIFIED_JSONL, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading classified posts"):
            if not line.strip():
                continue
            try:
                post = json.loads(line)
            except Exception:
                continue

            total_posts += 1

            # core fields
            uri           = post.get("uri", "")
            author_did    = post.get("author_did", "unknown")
            author_handle = post.get("author_handle", "unknown")
            month         = post.get("month", "unknown")
            text          = post.get("text", "")
            stance        = post.get("stance", "")
            stance_conf   = float(post.get("stance_conf", 0.5))
            weight        = float(post.get("weight", 1.0))

            # regimes — stored as pipe-separated string in classified file
            regimes_str = post.get("regimes", "Other_Unclear")
            regimes = [r.strip() for r in regimes_str.split("|") if r.strip()]
            if not regimes:
                regimes = ["Other_Unclear"]

            #part 1: audit buckets
            # Each regime a post belongs to gets it added to that regime's bucket, so audit samples are drawn per-regime accurately for multi-label posts
            for r in regimes:
                conf = stance_conf
                band = "high" if conf >= 0.8 else ("mid" if conf >= 0.5 else "low")
                audit_buckets[r][band].append({
                    "uri":         uri,
                    "regime":      r,
                    "all_regimes": regimes_str,
                    "stance":      stance,
                    "stance_conf": stance_conf,
                    "month":       month,
                    "weight":      weight,
                    "text":        text,
                })

            # parts 2-3: term counts
            # Tokens are accumulated for every regime a post is assigned to, not just the primary one
            tokens = tokenize(text)

            corpus_term_counts.update(tokens)
            corpus_total_tokens += len(tokens)

            for r in regimes:
                regime_term_counts[r].update(tokens)
                regime_total_tokens[r] += len(tokens)
                regime_month_term_counts[r][month].update(tokens)
                regime_month_total_tokens[r][month] += len(tokens)

            # part 4: author counts
            author_handles[author_did] = author_handle
            for r in regimes:
                author_regime_counts[author_did][r] += 1
            author_stance_counts[author_did][stance] += 1
            author_total_counts[author_did] += 1

    print(f"\n  Total classified posts read: {total_posts:,}")
    print(f"  Total corpus tokens: {corpus_total_tokens:,}")
    print(f"  Unique authors: {len(author_total_counts):,}")

    return {
        "audit_buckets":               audit_buckets,
        "corpus_term_counts":          corpus_term_counts,
        "corpus_total_tokens":         corpus_total_tokens,
        "regime_term_counts":          regime_term_counts,
        "regime_total_tokens":         regime_total_tokens,
        "regime_month_term_counts":    regime_month_term_counts,
        "regime_month_total_tokens":   regime_month_total_tokens,
        "author_regime_counts":        author_regime_counts,
        "author_stance_counts":        author_stance_counts,
        "author_total_counts":         author_total_counts,
        "author_handles":              author_handles,
    }

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# part 1: Regime audit sample


def build_audit_sample(data):
    """
    For each regime, draws posts from the high / mid / low confidence
    buckets in 40/35/25 proportion. Tops up from any remaining bucket
    if a band is too small to hit the per-regime target.
    """
    print("\n" + "="*60)
    print("PART 1: Building regime audit sample")
    print("="*60)

    audit_buckets = data["audit_buckets"]

    band_targets = {
        "high": int(AUDIT_SAMPLE_N * 0.40),
        "mid":  int(AUDIT_SAMPLE_N * 0.35),
        "low":  int(AUDIT_SAMPLE_N * 0.25),
    }

    all_rows = []

    for regime in REGIME_LABELS:
        b = audit_buckets.get(regime, {"high": [], "mid": [], "low": []})
        sampled = []

        for band, target in band_targets.items():
            pool = b[band]
            k = min(target, len(pool))
            sampled.extend(random.sample(pool, k) if k > 0 else [])

        # top up if bands were too small to hit AUDIT_SAMPLE_N
        if len(sampled) < AUDIT_SAMPLE_N:
            already_uris = {s.get("uri") for s in sampled}
            all_posts = [p for band_posts in b.values() for p in band_posts
                         if p.get("uri") not in already_uris]
            shortfall = AUDIT_SAMPLE_N - len(sampled)
            k = min(shortfall, len(all_posts))
            if k > 0:
                sampled.extend(random.sample(all_posts, k))

        n_high = sum(1 for p in sampled if float(p.get("stance_conf", 0.5)) >= 0.8)
        n_mid  = sum(1 for p in sampled if 0.5 <= float(p.get("stance_conf", 0.5)) < 0.8)
        n_low  = sum(1 for p in sampled if float(p.get("stance_conf", 0.5)) < 0.5)

        for post in sampled:
            conf = float(post.get("stance_conf", 0.5))
            all_rows.append({
                "regime":        post.get("regime", regime),
                "all_regimes":   post.get("all_regimes", regime),
                "stance":        post.get("stance", ""),
                "month":         post.get("month", ""),
                "stance_conf":   round(conf, 3),
                "conf_band":     "high" if conf >= 0.8 else ("mid" if conf >= 0.5 else "low"),
                "weight":        round(float(post.get("weight", 1.0)), 3),
                "text":          post.get("text", ""),
                "uri":           post.get("uri", ""),
                # Blank columns for your manual review
                "audit_correct": "",
                "audit_notes":   "",
            })

        print(f"  {regime}: {len(sampled)} posts "
              f"(high={n_high}, mid={n_mid}, low={n_low})")

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(AUDIT_DIR, "regime_audit_sample.csv")
    out.to_csv(out_path, index=False, quoting=1)
    wrote(out_path)
    print(f"\n  Total posts in audit sample: {len(out)}")
    print("  'audit_correct' and 'audit_notes' columns are blank for your review.")
    print("  'all_regimes' shows all predicted regimes for multi-label posts.")
    print("  To reduce to fewer posts: filter by conf_band or regime — no need to rerun.")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# part 2: Keyness per regime

def build_keyness(data):

    print("\n" + "="*60)
    print("PART 2: Keyness analysis per regime")
    print("="*60)

    corpus_counts = data["corpus_term_counts"]
    corpus_total  = data["corpus_total_tokens"]
    summary_rows  = []

    for regime in REGIME_LABELS:
        regime_counts = data["regime_term_counts"].get(regime, Counter())
        regime_total  = data["regime_total_tokens"].get(regime, 0)

        if regime_total == 0:
            print(f"  {regime}: no tokens, skipping")
            continue

        kdf = compute_keyness(
            regime_counts, regime_total,
            corpus_counts, corpus_total,
            KEYNESS_TOP_N
        )

        if kdf.empty:
            print(f"  {regime}: no keyness results")
            continue

        out_path = os.path.join(KEYNESS_DIR, f"keyness_{regime}.csv")
        kdf.to_csv(out_path, index=False)
        wrote(out_path)

        kdf.insert(0, "regime", regime)
        summary_rows.append(kdf)

        top5 = ", ".join(kdf["term"].head(5).tolist())
        print(f"  {regime}: top 5 — {top5}")

    if summary_rows:
        summary      = pd.concat(summary_rows, ignore_index=True)
        summary_path = os.path.join(KEYNESS_DIR, "keyness_all_regimes.csv")
        summary.to_csv(summary_path, index=False)
        wrote(summary_path)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# part 3: Temporal term trends


def build_temporal_terms(data):

    print("\n" + "="*60)
    print("PART 3: Temporal term trends per regime")
    print("="*60)

    corpus_counts = data["corpus_term_counts"]
    corpus_total  = data["corpus_total_tokens"]

    for regime in REGIME_LABELS:
        month_term_counts = data["regime_month_term_counts"].get(regime, {})
        month_totals      = data["regime_month_total_tokens"].get(regime, {})

        if not month_term_counts:
            print(f"  {regime}: no monthly data, skipping")
            continue

        months   = sorted(month_term_counts.keys())
        all_rows = []

        for month in months:
            m_counts = month_term_counts[month]
            m_total  = month_totals.get(month, 0)
            if m_total == 0:
                continue

            kdf = compute_keyness(
                m_counts, m_total,
                corpus_counts, corpus_total,
                TEMPORAL_TOP_N
            )
            if kdf.empty:
                continue

            kdf.insert(0, "month", month)
            kdf.insert(0, "regime", regime)
            all_rows.append(kdf)

        if not all_rows:
            continue

        out_df   = pd.concat(all_rows, ignore_index=True)
        out_path = os.path.join(TEMPORAL_DIR, f"temporal_terms_{regime}.csv")
        out_df.to_csv(out_path, index=False)
        wrote(out_path)

        print(f"  {regime}: {len(months)} months saved")

        # plot: top 10 terms by cumulative log-likelihood across all months
        top_terms = (
            out_df.groupby("term")["log_likelihood"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )

        pivot = out_df[out_df["term"].isin(top_terms)].pivot_table(
            index="month", columns="term", values="log_likelihood", fill_value=0.0
        ).sort_index()

        if pivot.empty or pivot.shape[0] < 2:
            continue

        plot_key_temporal = f"temporal_terms_{regime}"
        visible_terms = apply_regime_filter(list(pivot.columns), plot_key_temporal)
        pivot_plot = pivot[visible_terms] if visible_terms else pivot

        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(13, 6))
        for i, term in enumerate(pivot_plot.columns):
            plt.plot(pivot_plot.index, pivot_plot[term], label=term,
                     color=default_colors[i % len(default_colors)])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Month")
        plt.ylabel("Log-likelihood keyness")
        plt.title(f"Top term keyness over time — {display_name(regime)}")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(TEMPORAL_DIR, f"temporal_terms_{regime}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        wrote(plot_path)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# part 4: Prolific author robustness check


def build_prolific_author_check(data):

    print("\n" + "="*60)
    print("PART 4: Prolific author robustness check")
    print("="*60)

    author_total   = data["author_total_counts"]
    author_regime  = data["author_regime_counts"]
    author_stance  = data["author_stance_counts"]
    author_handles = data["author_handles"]

    if not author_total:
        print("  No author data — skipping.")
        return

    total_posts = sum(author_total.values())
    top_authors = [did for did, _ in author_total.most_common(PROLIFIC_TOP_N)]
    top_posts   = sum(author_total[did] for did in top_authors)
    top_pct     = top_posts / total_posts * 100 if total_posts else 0

    print(f"\n  Total classified posts: {total_posts:,}")
    print(f"  Top {PROLIFIC_TOP_N} authors: {top_posts:,} posts ({top_pct:.1f}% of corpus)")

    # save top author list with per-regime and per-stance breakdown
    author_rows = []
    for did in top_authors:
        row = {
            "author_did":    did,
            "author_handle": author_handles.get(did, did),
            "total_posts":   author_total[did],
            "pct_of_corpus": round(author_total[did] / total_posts * 100, 3),
        }
        for r in REGIME_LABELS:
            row[f"regime_{r}"] = author_regime[did].get(r, 0)
        for s in STANCE_LABELS:
            row[f"stance_{s}"] = author_stance[did].get(s, 0)
        author_rows.append(row)

    authors_df   = pd.DataFrame(author_rows)
    authors_path = os.path.join(ROBUST_DIR, f"top_{PROLIFIC_TOP_N}_authors.csv")
    authors_df.to_csv(authors_path, index=False)
    wrote(authors_path)

    # regime distribution: all vs. excluding top authors
    regime_all  = Counter()
    regime_excl = Counter()
    stance_all  = Counter()
    stance_excl = Counter()

    for did in author_total:
        for r, n in author_regime[did].items():
            regime_all[r] += n
            if did not in top_authors:
                regime_excl[r] += n
        for s, n in author_stance[did].items():
            stance_all[s] += n
            if did not in top_authors:
                stance_excl[s] += n

    total_all  = sum(regime_all.values())
    total_excl = sum(regime_excl.values())

    # Regime comparison table
    regime_rows = []
    for regime in REGIME_LABELS:
        n_all    = regime_all.get(regime, 0)
        n_excl   = regime_excl.get(regime, 0)
        s_all    = n_all  / total_all  * 100 if total_all  else 0
        s_excl   = n_excl / total_excl * 100 if total_excl else 0
        regime_rows.append({
            "regime":             regime,
            "share_all_%":        round(s_all,  2),
            "share_excl_top20_%": round(s_excl, 2),
            "delta_%pts":         round(s_excl - s_all, 2),
            "n_all":              n_all,
            "n_excl_top20":       n_excl,
        })

    regime_comp = pd.DataFrame(regime_rows)
    regime_comp_path = os.path.join(
        ROBUST_DIR, "regime_distribution_with_vs_without_top_authors.csv"
    )
    regime_comp.to_csv(regime_comp_path, index=False)
    wrote(regime_comp_path)

    # stance comparison table
    stance_rows = []
    total_stance_all  = sum(stance_all.values())
    total_stance_excl = sum(stance_excl.values())
    for stance in STANCE_LABELS:
        n_all  = stance_all.get(stance, 0)
        n_excl = stance_excl.get(stance, 0)
        s_all  = n_all  / total_stance_all  * 100 if total_stance_all  else 0
        s_excl = n_excl / total_stance_excl * 100 if total_stance_excl else 0
        stance_rows.append({
            "stance":             stance,
            "share_all_%":        round(s_all,  2),
            "share_excl_top20_%": round(s_excl, 2),
            "delta_%pts":         round(s_excl - s_all, 2),
            "n_all":              n_all,
            "n_excl_top20":       n_excl,
        })

    stance_comp = pd.DataFrame(stance_rows)
    stance_comp_path = os.path.join(
        ROBUST_DIR, "stance_distribution_with_vs_without_top_authors.csv"
    )
    stance_comp.to_csv(stance_comp_path, index=False)
    wrote(stance_comp_path)

    # print largest regime shifts
    print("\n  Regime share change when top authors removed:")
    for _, row in regime_comp.sort_values("delta_%pts", key=abs, ascending=False).iterrows():
        direction = "up" if row["delta_%pts"] > 0 else "down"
        print(f"    {row['regime']}: "
              f"{row['share_all_%']:.1f}% -> {row['share_excl_top20_%']:.1f}% "
              f"({row['delta_%pts']:+.2f}pp {direction})")

    # print stance shifts
    print("\n  Stance share change when top authors removed:")
    for _, row in stance_comp.sort_values("delta_%pts", key=abs, ascending=False).iterrows():
        direction = "up" if row["delta_%pts"] > 0 else "down"
        print(f"    {row['stance']}: "
              f"{row['share_all_%']:.1f}% -> {row['share_excl_top20_%']:.1f}% "
              f"({row['delta_%pts']:+.2f}pp {direction})")

    # side-by-side bar chart for regimes
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    visible_regimes = apply_regime_filter(REGIME_LABELS, "regime_robustness_comparison")
    regimes_short   = [display_name(r) for r in visible_regimes]
    bar_colors      = [label_color(r) or "steelblue" for r in visible_regimes]

    axes[0].barh(regimes_short,
                 [regime_all.get(r, 0) / total_all * 100 for r in visible_regimes],
                 color=bar_colors)
    axes[0].set_title("All authors")
    axes[0].set_xlabel("Share of posts (%)")

    axes[1].barh(regimes_short,
                 [regime_excl.get(r, 0) / total_excl * 100 for r in visible_regimes],
                 color=bar_colors)
    axes[1].set_title(f"Excluding top {PROLIFIC_TOP_N} authors")
    axes[1].set_xlabel("Share of posts (%)")

    plt.suptitle(
        f"Regime distribution: all authors vs. excluding top {PROLIFIC_TOP_N}",
        fontsize=12
    )
    plt.tight_layout()
    regime_plot_path = os.path.join(ROBUST_DIR, "regime_robustness_comparison.png")
    plt.savefig(regime_plot_path, dpi=200)
    plt.close()
    wrote(regime_plot_path)

    # side-by-side bar chart for stance
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    visible_stances = apply_regime_filter(STANCE_LABELS, "stance_robustness_comparison")
    stance_colors   = [label_color(s) or "steelblue" for s in visible_stances]

    axes[0].bar(visible_stances,
                [stance_all.get(s, 0) / total_stance_all * 100 for s in visible_stances],
                color=stance_colors)
    axes[0].set_title("All authors")
    axes[0].set_ylabel("Share of posts (%)")

    axes[1].bar(visible_stances,
                [stance_excl.get(s, 0) / total_stance_excl * 100 for s in visible_stances],
                color=stance_colors)
    axes[1].set_title(f"Excluding top {PROLIFIC_TOP_N} authors")

    plt.suptitle(
        f"Stance distribution: all authors vs. excluding top {PROLIFIC_TOP_N}",
        fontsize=12
    )
    plt.tight_layout()
    stance_plot_path = os.path.join(ROBUST_DIR, "stance_robustness_comparison.png")
    plt.savefig(stance_plot_path, dpi=200)
    plt.close()
    wrote(stance_plot_path)

    print(f"\n  Interpretation guide for delta column:")
    print(f"    < 2pp  — finding is robust to prolific author influence")
    print(f"    2-5pp  — modest influence, worth a footnote")
    print(f"    > 5pp  — notable influence, warrants a limitations paragraph")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main

def main():

    print("06_audit_and_analyze.py")
    print(f"Work directory: {WORK_DIR}")

    if not os.path.exists(CLASSIFIED_JSONL):
        print(f"\nERROR: classified_posts.jsonl not found at {CLASSIFIED_JSONL}")
        print("Run 04_apply_models_streaming.py first.")
        return

    # single streaming pass — collects everything for all four parts
    data = streaming_pass_classified()

    if data is None:
        return

    build_audit_sample(data)
    build_keyness(data)
    build_temporal_terms(data)
    build_prolific_author_check(data)

    print(f"\n{'='*60}")
    print(f"All outputs written ({len(_written)} files):")
    for p in _written:
        print(f"  {p}")

if __name__ == "__main__":
    main()