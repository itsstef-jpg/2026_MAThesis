
"""
Generates the corpus-level analysis tables and figures.

This script reads the trends CSV produced by 04_apply_models.py
and writes per-regime trends over time, raw and engagement-weighted volume
charts, stance x regime cross-tabulations, spike tables, engagement
amplification ratios, temporal co-attention between regimes, and
several corpus-level summary tables.

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import ujson as json
except ImportError:
    import json
from config import WORK_DIR, SPIKE_Z, REGIME_LABELS, TOTAL_CORPUS_POSTS, DATA_PATH

TRENDS   = os.path.join(WORK_DIR, "monthly_trends_supervised.csv")
TOPJSONL = os.path.join(WORK_DIR, "top_posts_by_month_stance_regime.jsonl")

VIZ_DIR = os.path.join(WORK_DIR, "viz")
os.makedirs(VIZ_DIR, exist_ok=True)

_written = []

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    _written.append(path)
    print("Wrote:", path)

def wrote(path):
    _written.append(path)
    print("Wrote:", path)

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

# per-plot regime/stance filter settings.
# "include": [...] — only show these labels
# "exclude": [...] — show all labels except these

# Use internal REGIME_LABELS names (not display names) as values.
PLOT_REGIME_FILTERS = {
}

def apply_regime_filter(pivot, plot_key):
    f = PLOT_REGIME_FILTERS.get(plot_key)
    if f is None:
        return pivot
    if "include" in f:
        keep = [c for c in pivot.columns if c in f["include"]]
        return pivot[keep] if keep else pivot
    if "exclude" in f:
        keep = [c for c in pivot.columns if c not in f["exclude"]]
        return pivot[keep] if keep else pivot
    return pivot

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Plotting helpers

def plot_kind(df, kind, out_png, title, top_n=None, plot_key=None):
 # line chart of egagement weighted shares per month, one line per label   
    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip plot: no rows for kind={kind}")
        return

    pivot = sub.pivot(index="month", columns="label", values="share").fillna(0.0).sort_index()

    if top_n is not None and pivot.shape[1] > top_n:
        top_cols = pivot.sum(axis=0).sort_values(ascending=False).head(top_n).index
        pivot = pivot[top_cols]

    # apply per-plot include/exclude filter
    if plot_key:
        pivot = apply_regime_filter(pivot, plot_key)
    if pivot.empty:
        print(f"Skip plot: all labels filtered out for plot_key={plot_key}")
        return

    plt.figure(figsize=(12, 6))
    for c in pivot.columns:
        plt.plot(pivot.index, pivot[c],
                 label=display_name(c),
                 color=label_color(c))
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Share of monthly attention (engagement-weighted)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    savefig(out_png)

def plot_kind_counts(df, kind, out_png, title, top_n=None, plot_key=None):
 # bar chart for raw post counts per label per month 

    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip count plot: no rows for kind={kind}")
        return
    if "post_count" not in sub.columns or sub["post_count"].sum() == 0:
        print(f"Skip count plot: no post_count data for kind={kind} "
              f"(re-run 04_ to generate counts)")
        return

    pivot = sub.pivot(index="month", columns="label", values="post_count"
                      ).fillna(0.0).sort_index()

    if top_n is not None and pivot.shape[1] > top_n:
        top_cols = pivot.sum(axis=0).sort_values(ascending=False).head(top_n).index
        pivot = pivot[top_cols]

    if plot_key:
        pivot = apply_regime_filter(pivot, plot_key)
    if pivot.empty:
        print(f"Skip count plot: all labels filtered for plot_key={plot_key}")
        return

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        label_color(c) or default_colors[i % len(default_colors)]
        for i, c in enumerate(pivot.columns)
    ]
    pivot.plot(kind="bar", stacked=True, figsize=(13, 6),
            color=colors, width=0.8)    
    
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Number of posts")
    plt.title(title)
    plt.legend(
        [display_name(c) for c in pivot.columns],
        loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1)
    )
    savefig(out_png)


def plot_kind_overlaid(df, kind, out_png, title, top_n=None, plot_key=None):
 # stacked bars for raw counts + engagement weighted lines overlaid   

    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip overlaid plot: no rows for kind={kind}")
        return
    if "post_count" not in sub.columns or sub["post_count"].sum() == 0:
        print(f"Skip overlaid plot: no post_count data for kind={kind} "
              f"(re-run 04_ to generate counts)")
        return

    pivot_n = sub.pivot(index="month", columns="label", values="post_count"
                        ).fillna(0.0).sort_index()
    pivot_s = sub.pivot(index="month", columns="label", values="share"
                        ).fillna(0.0).sort_index()

    if top_n is not None and pivot_n.shape[1] > top_n:
        top_cols = pivot_n.sum(axis=0).sort_values(ascending=False).head(top_n).index
        pivot_n = pivot_n[top_cols]
        pivot_s = pivot_s[[c for c in top_cols if c in pivot_s.columns]]

    if plot_key:
        pivot_n = apply_regime_filter(pivot_n, plot_key)
        pivot_s = apply_regime_filter(pivot_s, plot_key)
    if pivot_n.empty:
        print(f"Skip overlaid plot: all labels filtered for plot_key={plot_key}")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        label_color(c) or default_colors[i % len(default_colors)]
        for i, c in enumerate(pivot_n.columns)
    ]
    
    # stacked bars on left axis (raw counts)
    bottom = np.zeros(len(pivot_n))
    for col, color in zip(pivot_n.columns, colors):
        ax1.bar(range(len(pivot_n)), pivot_n[col].values,
                bottom=bottom, color=color, alpha=0.45,
                label=f"{display_name(col)} (n)")
        bottom += pivot_n[col].values

    # share lines on right axis (engagement-weighted)
    for col in pivot_s.columns:
        ax2.plot(range(len(pivot_s)), pivot_s[col].values,
                 color=label_color(col), linewidth=2,
                 linestyle="--", marker="o", markersize=3,
                 label=f"{display_name(col)} (share)")

    ax1.set_xticks(range(len(pivot_n)))
    ax1.set_xticklabels(pivot_n.index, rotation=45, ha="right")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Post count", color="black")
    ax2.set_ylabel("Engagement-weighted share", color="black")
    plt.title(title)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left",
               fontsize=7, bbox_to_anchor=(1.05, 1))

    savefig(out_png)


def plot_kind_separate(df, kind, out_bar_png, out_line_png, title_bar, title_line,
                        top_n=None, plot_key=None):
 # raw count bar chart and engagement weighted share line chart

    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip separate plots: no rows for kind={kind}")
        return
    if "post_count" not in sub.columns or sub["post_count"].sum() == 0:
        print(f"Skip separate bar plot: no post_count data for kind={kind} "
              f"(re-run 04_ to generate counts)")
        # still attempt the share lines even if counts are missing
        out_bar_png = None

    pivot_n = sub.pivot(index="month", columns="label", values="post_count"
                        ).fillna(0.0).sort_index() if out_bar_png else None
    pivot_s = sub.pivot(index="month", columns="label", values="share"
                        ).fillna(0.0).sort_index()

    # restrict to top_n labels by total count
    if top_n is not None:
        if pivot_n is not None and pivot_n.shape[1] > top_n:
            top_cols = pivot_n.sum(axis=0).sort_values(ascending=False).head(top_n).index
            pivot_n = pivot_n[top_cols]
            pivot_s = pivot_s[[c for c in top_cols if c in pivot_s.columns]]
        elif pivot_n is None and pivot_s.shape[1] > top_n:
            top_cols = pivot_s.sum(axis=0).sort_values(ascending=False).head(top_n).index
            pivot_s = pivot_s[top_cols]

    if plot_key:
        if pivot_n is not None:
            pivot_n = apply_regime_filter(pivot_n, plot_key)
        pivot_s = apply_regime_filter(pivot_s, plot_key)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # bar chart raw counts
    if pivot_n is not None and not pivot_n.empty and out_bar_png:
        colors = [
            label_color(c) or default_colors[i % len(default_colors)]
            for i, c in enumerate(pivot_n.columns)
        ]
        pivot_n.plot(kind="bar", stacked=True, figsize=(13, 6),
                     color=colors, width=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Month")
        plt.ylabel("Number of posts")
        plt.title(title_bar)
        plt.legend(
            [display_name(c) for c in pivot_n.columns],
            loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1)
        )
        savefig(out_bar_png)

    # line chart engagement-weighted shares
    if not pivot_s.empty:
        plt.figure(figsize=(12, 6))
        for c in pivot_s.columns:
            plt.plot(pivot_s.index, pivot_s[c],
                     label=display_name(c),
                     color=label_color(c))
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Month")
        plt.ylabel("Share of monthly attention (engagement-weighted)")
        plt.title(title_line)
        plt.legend(loc="best", fontsize=8)
        savefig(out_line_png)


def export_summary_table(df, out_csv, out_md):
 # single flat summary table convering every regime and stance
 # raw share instance - sum to 100%
 # raw share corpus - sum to >100% due to multi-labelling
 # eng share instance - regime engagement / total regime engagement
 # eng share corpus - regime engagement / corpus-corrected total 

    rows = []
    for kind in ("stance", "regime"):
        sub = df[df["kind"] == kind].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby("label")[["post_count", "weight_sum"]]
            .sum()
            .reset_index()
            .rename(columns={"post_count": "total_posts", "weight_sum": "total_weight"})
        )
        tot_label_posts  = agg["total_posts"].sum()
        tot_label_weight = agg["total_weight"].sum()

        if kind == "regime":
            mean_labels = tot_label_posts / TOTAL_CORPUS_POSTS if TOTAL_CORPUS_POSTS else 1.0
            tot_corpus_weight = tot_label_weight / mean_labels if mean_labels else tot_label_weight

            agg["raw_share_instance"] = (
                agg["total_posts"] / tot_label_posts if tot_label_posts else 0.0
            )
            agg["raw_share_corpus"] = (
                agg["total_posts"] / TOTAL_CORPUS_POSTS if TOTAL_CORPUS_POSTS else 0.0
            )
            agg["eng_share_instance"] = (
                agg["total_weight"] / tot_label_weight if tot_label_weight else 0.0
            )
            agg["eng_share_corpus"] = (
                agg["total_weight"] / tot_corpus_weight if tot_corpus_weight else 0.0
            )
        else:
            agg["raw_share"] = agg["total_posts"]  / tot_label_posts  if tot_label_posts  else 0.0
            agg["eng_share"] = agg["total_weight"] / tot_label_weight if tot_label_weight else 0.0

        agg["kind"] = kind
        agg["display_label"] = agg["label"].map(display_name)
        rows.append(agg)

    if not rows:
        print("Skip summary table: no stance or regime rows found.")
        return

    summary = pd.concat(rows, ignore_index=True)
    summary = summary.sort_values(
        ["kind", "total_posts"], ascending=[True, False]
    ).reset_index(drop=True)

    out_cols = ["kind", "label", "display_label", "total_posts", "total_weight",
                "raw_share_instance", "raw_share_corpus",
                "eng_share_instance", "eng_share_corpus",
                "raw_share", "eng_share"]
    out_cols = [c for c in out_cols if c in summary.columns]
    summary[out_cols].to_csv(out_csv, index=False, float_format="%.4f")
    wrote(out_csv)


    for kind in ("regime", "stance"):
        sub = summary[summary["kind"] == kind]
        if sub.empty:
            continue
        md_lines.append(f"## {kind.title()} summary")
        md_lines.append("")

        if kind == "regime":
            md_lines.append(
                "| Label | Posts | Engagement | "
                "Corpus raw share | Corpus eng. share | "
                "Instance raw share | Instance eng. share |"
            )
            md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                md_lines.append(
                    f"| {r['display_label']} "
                    f"| {int(r['total_posts']):,} "
                    f"| {int(r['total_weight']):,} "
                    f"| {r['raw_share_corpus']*100:.1f}% "
                    f"| {r['eng_share_corpus']*100:.1f}% "
                    f"| {r['raw_share_instance']*100:.1f}% "
                    f"| {r['eng_share_instance']*100:.1f}% |"
                )
            md_lines.extend([
                "",
                (f"Corpus share denominator: {TOTAL_CORPUS_POSTS:,} unique posts. "),
                "Instance share denominator: total label instances (multi-label inflated). "
                "Retained for transparency.",
            ])
        else:
            md_lines.append(
                "| Label | Posts | Engagement | Raw share | Eng. share |"
            )
            md_lines.append("|---|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                md_lines.append(
                    f"| {r['display_label']} "
                    f"| {int(r['total_posts']):,} "
                    f"| {int(r['total_weight']):,} "
                    f"| {r['raw_share']*100:.1f}% "
                    f"| {r['eng_share']*100:.1f}% |"
                )
        md_lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    wrote(out_md)

    return summary


def export_per_regime_stance(df, out_csv, out_md):
 # for each regime, compute the stance composition as a share of that regime's own posts (not as a share of the full corpus)
 # multi-label: a post carrying to regime labels is counted in each regime's denominator independently, so shares are only internally consistent

    sx = df[df["kind"] == "stance_x_regime"].copy()
    if sx.empty:
        print("Skip per-regime stance table: no stance_x_regime rows found.")
        return

    # parse "stance__regime" label into two columns
    sx[["stance", "regime"]] = sx["label"].str.split("__", expand=True)

    # aggregate across all months
    agg = (
        sx.groupby(["regime", "stance"])[["post_count", "weight_sum"]]
        .sum()
        .reset_index()
    )

    # regime totals
    regime_totals = (
        agg.groupby("regime")[["post_count", "weight_sum"]]
        .sum()
        .rename(columns={"post_count": "regime_posts", "weight_sum": "regime_weight"})
    )
    agg = agg.merge(regime_totals, on="regime")
    agg["raw_share"]  = agg["post_count"] / agg["regime_posts"]
    agg["eng_share"]  = agg["weight_sum"] / agg["regime_weight"]

    stances = sorted(agg["stance"].unique())

    # build one row per regime
    regimes_ordered = [r for r in REGIME_LABELS if r in agg["regime"].unique()]

    wide_rows = []
    for regime in regimes_ordered:
        sub = agg[agg["regime"] == regime]
        row = {
            "regime": regime,
            "display_label": display_name(regime),
            "total_posts": int(sub["regime_posts"].iloc[0]) if not sub.empty else 0,
            "total_weight": int(sub["regime_weight"].iloc[0]) if not sub.empty else 0,
        }
        for stance in stances:
            s = sub[sub["stance"] == stance]
            row[f"raw_{stance}"] = float(s["raw_share"].iloc[0]) if not s.empty else 0.0
            row[f"eng_{stance}"] = float(s["eng_share"].iloc[0]) if not s.empty else 0.0
            row[f"n_{stance}"]   = int(s["post_count"].iloc[0]) if not s.empty else 0
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)

    # CSV: all columns
    wide.to_csv(out_csv, index=False, float_format="%.4f")
    wrote(out_csv)

    # Markdown: raw share columns only
    md_lines = [
        "# Per-regime stance composition",
        "",
        ("Stance shares are computed within each regime: each value is the fraction "
         "of that regime's posts carrying that stance label. Values in each row sum "
         "to approximately 100%. Because regime is multi-label, a post counted in two "
         "regimes contributes to each regime's denominator independently."),
        "",
        "## Raw post-count shares",
        "",
    ]

    # header: Label | total posts | one column per stance
    header = "| Regime | Posts | " + " | ".join(s.title() for s in stances) + " |"
    sep    = "|---|---:|" + "|".join(["---:" for _ in stances]) + "|"
    md_lines.extend([header, sep])

    for _, r in wide.iterrows():
        stance_cells = " | ".join(
            f"{r[f'raw_{s}']*100:.1f}%" for s in stances
        )
        md_lines.append(
            f"| {r['display_label']} | {int(r['total_posts']):,} | {stance_cells} |"
        )

    md_lines.extend([
        "",
        "## Engagement-weighted shares",
        "",
        header.replace("Posts", "Engagement"),
        sep,
    ])

    for _, r in wide.iterrows():
        stance_cells = " | ".join(
            f"{r[f'eng_{s}']*100:.1f}%" for s in stances
        )
        md_lines.append(
            f"| {r['display_label']} | {int(r['total_weight']):,} | {stance_cells} |"
        )

    md_lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    wrote(out_md)

    return wide


def plot_stance_regime_segmented(df, out_png, title,
                                  use_counts=True,
                                  regime_filter=None,
                                  plot_key="stance_regime_segmented"):
 # grouped stacked bar chart - one group per month, one bar per stance, each bar subdivided by regime  

    sx = df[df["kind"] == "stance_x_regime"].copy()
    if sx.empty:
        print("Skip segmented plot: no stance_x_regime rows found")
        return

    val_col = "post_count" if use_counts else "weight_sum"
    if use_counts and ("post_count" not in sx.columns or sx["post_count"].sum() == 0):
        print("Skip segmented plot: no post_count data "
              "(re-run 04_ to generate counts)")
        return

    # parse the combined label "stance__regime" back into two columns
    sx[["stance", "regime"]] = sx["label"].str.split("__", expand=True)

    # determine which regimes to show
    if regime_filter is not None:
        visible_regimes = regime_filter
    else:
        # use PLOT_REGIME_FILTERS if set, otherwise default to all except Other_Unclear
        f = PLOT_REGIME_FILTERS.get(plot_key)
        if f and "include" in f:
            visible_regimes = f["include"]
        elif f and "exclude" in f:
            visible_regimes = [r for r in REGIME_LABELS if r not in f["exclude"]]
        else:
            visible_regimes = [r for r in REGIME_LABELS if r != "Other_Unclear"]

    sx = sx[sx["regime"].isin(visible_regimes)]
    if sx.empty:
        print("Skip segmented plot: all regimes filtered out")
        return

    months  = sorted(sx["month"].unique())
    stances = sorted(sx["stance"].unique())

    # bar layout: groups = months, within each group one bar per stance
    n_months  = len(months)
    n_stances = len(stances)
    group_w   = 0.8 #total width of one month group
    bar_w     = group_w / n_stances # width of one stance bar
    group_gap = 1.0 # spacing between month groups

    fig, ax = plt.subplots(figsize=(max(14, n_months * 2.2), 7))

    # draw stacked bars for each month × stance combination
    for m_idx, month in enumerate(months):
        group_center = m_idx * group_gap
        for s_idx, stance in enumerate(stances):
            bar_x   = group_center + (s_idx - n_stances / 2 + 0.5) * bar_w
            sub     = sx[(sx["month"] == month) & (sx["stance"] == stance)]
            bottom  = 0.0

            for regime in visible_regimes:
                row = sub[sub["regime"] == regime]
                val = float(row[val_col].iloc[0]) if not row.empty else 0.0
                if val > 0:
                    ax.bar(bar_x, val, bar_w * 0.9,
                           bottom=bottom,
                           color=label_color(regime),
                           label=display_name(regime))
                    bottom += val

            # Label the stance below the bar group
            if m_idx == 0:
                ax.text(bar_x, -ax.get_ylim()[1] * 0.03,
                        stance, ha="center", va="top", fontsize=8, rotation=45)

    # month labels centered on each group
    ax.set_xticks([m_idx * group_gap for m_idx in range(n_months)])
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_xlabel("Month  (stance labels below each bar group)")
    ax.set_ylabel("Post count" if use_counts else "Engagement-weighted sum")
    ax.set_title(title)

    # deduplicated legend (regime colors only, no duplicates from loop)
    seen = {}
    for regime in visible_regimes:
        seen[display_name(regime)] = plt.Rectangle(
            (0, 0), 1, 1, color=label_color(regime)
        )
    ax.legend(seen.values(), seen.keys(),
              loc="upper left", fontsize=8,
              bbox_to_anchor=(1.01, 1), title="Regime")

    savefig(out_png)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Spike detection

def zscore_spikes(series: pd.Series, z=2.0):
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        sd = 1.0
    zs = (s - mu) / sd
    return zs[zs >= z].sort_values(ascending=False)

def spike_table_for_kind(df, kind, out_csv, z=2.0):
    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip spikes: no rows for kind={kind}")
        return

    pivot = sub.pivot(index="month", columns="label", values="share").fillna(0.0).sort_index()
    spike_rows = []
    for label in pivot.columns:
        spikes = zscore_spikes(pivot[label], z=z)
        for month, zval in spikes.items():
            spike_rows.append({
                "kind":   kind,
                "label":  label,
                "month":  month,
                "zscore": float(zval),
                "share":  float(pivot.loc[month, label]),
            })
    out = pd.DataFrame(spike_rows)
    if out.empty:
        print(f"  No spikes detected for kind={kind} at z>={z}")
        return
    out = out.sort_values("zscore", ascending=False)
    out.to_csv(out_csv, index=False)
    wrote(out_csv)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Raw counts export

def export_raw_counts(df, kind, out_csv):
    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip raw counts export: no rows for kind={kind}")
        return
    out = sub[["month", "label", "weight_sum"]].sort_values(
        ["month", "weight_sum"], ascending=[True, False]
    )
    out.to_csv(out_csv, index=False)
    wrote(out_csv)


# ----------------------------------------------------------------------
# Engagement amplification
# ----------------------------------------------------------------------
# ratio of per-post engagement to the corpus average per-post engagement, calculated separately for each class

"""
The amplification ratio for any class is:
amplification = (class_weight / class_posts) /
                (corpus_weight / corpus_posts)
                    = per-post engagement of class /
                    per-post engagement of corpus mean

- stance is single label per post - engagement weighted and raw count shares both sum to 100% of the corpus
- regime is multi label - two interpretations of share of corpus are possible - share of regime instance space (multi counted) or share of unique-corpus space (corrected). Both are reported.

Amplification ratio is identical regardless of which share denominator is used.
    """

def export_amplification(df, kind, out_csv, out_md):
    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip {kind} amplification: no rows for kind={kind}")
        return None

    agg = (
        sub.groupby("label")[["weight_sum", "post_count"]]
        .sum()
        .reset_index()
        .rename(columns={"label":      "class",
                         "weight_sum": "class_weight",
                         "post_count": "class_posts"})
    )

    total_label_weight = agg["class_weight"].sum()
    total_label_posts  = agg["class_posts"].sum()

    if kind == "regime":
        mean_labels_per_post = total_label_posts / TOTAL_CORPUS_POSTS
        total_corpus_weight  = total_label_weight / mean_labels_per_post
        agg["eng_share_direct"]    = agg["class_weight"] / total_label_weight
        agg["raw_share_direct"]    = agg["class_posts"]  / total_label_posts
        agg["eng_share_corrected"] = agg["class_weight"] / total_corpus_weight
        agg["raw_share_corrected"] = agg["class_posts"]  / TOTAL_CORPUS_POSTS
        eng_share_for_md = "eng_share_corrected"
        raw_share_for_md = "raw_share_corrected"
        out_cols = ["class", "class_weight", "class_posts",
                    "eng_share_direct", "raw_share_direct",
                    "eng_share_corrected", "raw_share_corrected",
                    "amplification"]
    else:
        agg["eng_share"] = agg["class_weight"] / total_label_weight
        agg["raw_share"] = agg["class_posts"]  / total_label_posts
        eng_share_for_md = "eng_share"
        raw_share_for_md = "raw_share"
        out_cols = ["class", "class_weight", "class_posts",
                    "eng_share", "raw_share", "amplification"]

    # amplification
    agg["amplification"] = (agg["class_weight"] / agg["class_posts"]) / (
        total_label_weight / total_label_posts
    )

    agg = agg.sort_values("amplification", ascending=False).reset_index(drop=True)

    agg[out_cols].to_csv(out_csv, index=False, float_format="%.4f")
    wrote(out_csv)

    md_lines = [
        f"| {kind.title()} | Amplification | Engagement share | Raw-count share |",
        "|---|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        md_lines.append(
            f"| {display_name(r['class'])} | {r['amplification']:.2f} | "
            f"{r[eng_share_for_md]*100:.1f}% | "
            f"{r[raw_share_for_md]*100:.1f}% |"
        )
    md_text = "\n".join(md_lines) + "\n"
    with open(out_md, "w") as f:
        f.write(md_text)
    wrote(out_md)

    return agg


def plot_amplification(amp_df, out_png, kind):
 # horizontal bar chart of amplification ratios per class, with 1.0 reference line   
    if amp_df is None or amp_df.empty:
        print(f"Skip {kind} amplification plot: no data.")
        return

    amp_df = amp_df.iloc[::-1].reset_index(drop=True)
    labels = [display_name(c) for c in amp_df["class"]]
    values = amp_df["amplification"].values
    colors = ["#2a9d8f" if v >= 1.0 else "#e76f51" for v in values]

    plt.figure(figsize=(10, max(3, 0.6 * len(labels) + 1.5)))
    bars = plt.barh(labels, values, color=colors, edgecolor="white")
    plt.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    plt.xlabel("Engagement amplification ratio (engagement share / volume share)")
    plt.title(
        f"Engagement amplification per {kind} class\n"
        "(>1.0 amplifying, <1.0 deflating)"
    )

    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val >= 1.0 else val - 0.02
        h_align = "left" if val >= 1.0 else "right"
        plt.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", ha=h_align, fontsize=9)

    plt.xlim(0, max(values) * 1.15)
    savefig(out_png)


# ----------------------------------------------------------------------
# Backwards-compatible aliases
# ----------------------------------------------------------------------
# Earlier versions of this script had regime-specific function names
# (`export_engagement_amplification`, `plot_engagement_amplification`).
# The aliases below preserve the older API for any external code or
# notebook that imports from this module by name.

def export_engagement_amplification(df, out_csv, out_md):
    return export_amplification(df, "regime", out_csv, out_md)

def plot_engagement_amplification(amp_df, out_png):
    return plot_amplification(amp_df, out_png, "regime")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Corpus-level summary statistics

def count_authors_from_jsonl(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"Skip author count: JSONL file not found at {jsonl_path}")
        return None

    authors = set()
    n_posts = 0
    n_no_author = 0

    print(f"Counting unique authors in {jsonl_path} (this may take a minute)...")
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    post = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                n_posts += 1

                # Field-priority pattern matching 04_v2_apply_models.py
                did = post.get("author_did") or post.get("did", "")
                if not did:
                    # Fallback: extract from AT URI
                    uri = post.get("uri", "")
                    if uri.startswith("at://"):
                        rest = uri[5:]  # strip "at://"
                        did = rest.split("/", 1)[0]

                if did:
                    authors.add(did)
                else:
                    n_no_author += 1
    except OSError as e:
        print(f"Skip author count: could not read JSONL ({e})")
        return None

    return {
        "n_unique_authors": len(authors),
        "n_posts_seen":     n_posts,
        "n_posts_no_author": n_no_author,
    }


def export_corpus_statistics(df, out_csv, out_md, author_stats=None):
    stance = df[df["kind"] == "stance"].copy()
    if stance.empty:
        print("Skip corpus statistics: no stance rows found.")
        return

    monthly = (
        stance.groupby("month")[["weight_sum", "post_count"]]
        .sum()
        .reset_index()
        .sort_values("month")
        .rename(columns={"weight_sum":  "monthly_engagement",
                         "post_count":  "monthly_posts"})
    )
    monthly["engagement_per_post"] = monthly["monthly_engagement"] / monthly["monthly_posts"]

    monthly.to_csv(out_csv, index=False, float_format="%.2f")
    wrote(out_csv)

    total_engagement = monthly["monthly_engagement"].sum()
    n_months         = len(monthly)
    first_month      = monthly["month"].iloc[0]
    last_month       = monthly["month"].iloc[-1]
    max_row          = monthly.loc[monthly["monthly_posts"].idxmax()]
    min_row          = monthly.loc[monthly["monthly_posts"].idxmin()]
    eng_per_post     = total_engagement / TOTAL_CORPUS_POSTS
    sum_monthly_posts = int(monthly["monthly_posts"].sum())

    # derive multi-label statistics from regime rows 
    regime = df[df["kind"] == "regime"].copy()
    if not regime.empty:
        total_regime_instances = int(regime["post_count"].sum())
        mean_labels_per_post = total_regime_instances / TOTAL_CORPUS_POSTS if TOTAL_CORPUS_POSTS else None
        multi_label_pct = None
        # approximate multi-label fraction: (instances - posts) / posts gives the average number of extra labels per post
    else:
        total_regime_instances = None
        mean_labels_per_post = None

    md_lines = [
        "## Corpus summary statistics",
        "",
        f"- **Total posts**: {TOTAL_CORPUS_POSTS:,} (from config; sum across monthly stance counts: {sum_monthly_posts:,})",
        f"- **Period**: {first_month} to {last_month} ({n_months} months)",
        f"- **Total engagement**: {total_engagement:,.0f}",
        f"- **Engagement per post (corpus mean)**: {eng_per_post:.2f}",
    ]
    if author_stats:
        md_lines.append(
            f"- **Unique authors**: {author_stats['n_unique_authors']:,} "
            f"(from {author_stats['n_posts_seen']:,} posts read; "
            f"{author_stats['n_posts_no_author']:,} posts had no recoverable author DID)"
        )
    else:
        md_lines.append("- **Unique authors**: not computed (run count_authors_from_jsonl to add)")
    md_lines.extend([
        f"- **Highest-volume month**: {max_row['month']} ({int(max_row['monthly_posts']):,} posts)",
        f"- **Lowest-volume month**: {min_row['month']} ({int(min_row['monthly_posts']):,} posts)",
    ])
    if total_regime_instances is not None and mean_labels_per_post is not None:
        md_lines.extend([
            "",
            "## Multi-label note",
            "",
            ("Regime classification is multi-label: each post may carry one or more regime "
             "labels. Regime post counts therefore sum to more than the total unique posts."),
            "",
            f"- **Total regime label instances**: {total_regime_instances:,}",
            f"- **Mean regime labels per post**: {mean_labels_per_post:.2f}",
            ("- **Implication for shares**: regime raw shares computed as "
             "label_instances / TOTAL_CORPUS_POSTS sum to more than 100% across regimes. "
             ),
        ])
    md_lines.extend([
        "",
        "## Monthly breakdown",
        "",
        "| Month | Posts | Engagement | Engagement per post |",
        "|---|---:|---:|---:|",
    ])
    for _, r in monthly.iterrows():
        md_lines.append(
            f"| {r['month']} | {int(r['monthly_posts']):,} | "
            f"{int(r['monthly_engagement']):,} | {r['engagement_per_post']:.2f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    with open(out_md, "w") as f:
        f.write(md_text)
    wrote(out_md)

    return monthly


def plot_corpus_volume(monthly, out_png):
    if monthly is None or monthly.empty:
        print("Skip corpus volume plot: no data.")
        return

    fig, ax1 = plt.subplots(figsize=(11, 5))
    months = monthly["month"].astype(str).tolist()
    posts  = monthly["monthly_posts"].values
    eng    = monthly["monthly_engagement"].values

    ax1.bar(months, posts, color="#457b9d", alpha=0.75, edgecolor="white", label="Posts")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Posts per month", color="#457b9d")
    ax1.tick_params(axis="y", labelcolor="#457b9d")
    ax1.tick_params(axis="x", rotation=45)
    for tick in ax1.get_xticklabels():
        tick.set_horizontalalignment("right")

    ax2 = ax1.twinx()
    ax2.plot(months, eng, color="#e76f51", marker="o", linewidth=2, label="Engagement")
    ax2.set_ylabel("Engagement (likes + reposts + replies)", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    plt.title("Monthly post volume and engagement")
    fig.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    _written.append(out_png)
    print("Wrote:", out_png)


def plot_corpus_volume_separate(monthly, out_bar_png, out_line_png):
    if monthly is None or monthly.empty:
        print("Skip corpus volume separate plots: no data.")
        return

    months = monthly["month"].astype(str).tolist()
    posts  = monthly["monthly_posts"].values
    eng    = monthly["monthly_engagement"].values

    # bar chart post counts
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(months, posts, color="#457b9d", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Month")
    ax.set_ylabel("Posts per month")
    ax.set_title("Monthly post volume")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    plt.savefig(out_bar_png, dpi=200)
    plt.close()
    _written.append(out_bar_png)
    print("Wrote:", out_bar_png)

    # line chart engagement totals
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(months, eng, color="#e76f51", marker="o", linewidth=2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Engagement (likes + reposts + replies)")
    ax.set_title("Monthly engagement total")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    plt.savefig(out_line_png, dpi=200)
    plt.close()
    _written.append(out_line_png)
    print("Wrote:", out_line_png)


def plot_corpus_summary_table(monthly, author_stats, out_png):
    if monthly is None or monthly.empty:
        print("Skip corpus summary table: no data.")
        return

    total_posts      = TOTAL_CORPUS_POSTS
    total_engagement = int(monthly["monthly_engagement"].sum())
    n_months         = len(monthly)
    n_authors        = author_stats["n_unique_authors"] if author_stats else None

    # layout: a 2x2 grid of statistic cards
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor("white")

    # each card: a big number on top, a small label below
    cards = [
        (axes[0, 0], f"{total_posts:,}",
         "posts in corpus",     "#457b9d"),
        (axes[0, 1], f"{n_authors:,}" if n_authors else "—",
         "unique authors",      "#2a9d8f"),
        (axes[1, 0], f"{n_months}",
         "months\n(Dec 2024 to Jan 2026)",   "#6a4c93"),
        (axes[1, 1], f"{total_engagement:,}",
         "total engagement\n(likes + reposts + replies)", "#e76f51"),
    ]
    for ax, big, small, color in cards:
        ax.text(0.5, 0.62, big, fontsize=26, fontweight="bold",
                ha="center", va="center", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.22, small, fontsize=11, ha="center", va="center",
                color="#333333", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(1)

    plt.suptitle("Corpus at a glance", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=200, facecolor="white")
    plt.close()
    _written.append(out_png)
    print("Wrote:", out_png)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Temporal co-attention of regimes

def build_regime_coattention_from_topjsonl(path):
 # cosine similarity between each regime's (month x stance) weight vector, measuring whether regimes attract attention in the same months and stances over time
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({
                "month":  obj.get("month"),
                "stance": obj.get("stance"),
                "regime": obj.get("regime"),
                "weight": float(obj.get("weight", 0.0)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    ms = df.groupby(["month", "stance", "regime"], as_index=False)["weight"].sum()
    pivot = ms.pivot_table(
        index=["month", "stance"], columns="regime", values="weight", fill_value=0.0
    )

    regimes = list(pivot.columns)
    X = pivot.values

    co = []
    for i, r1 in enumerate(regimes):
        for j in range(i + 1, len(regimes)):
            r2 = regimes[j]
            v1 = X[:, i]
            v2 = X[:, j]
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            co.append({"regime_a": r1, "regime_b": r2, "temporal_coattention": sim})

    return pd.DataFrame(co).sort_values("temporal_coattention", ascending=False)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Tabular export helpers

def export_kind(df, kind, out_csv):
    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        print(f"Skip export: no rows for kind={kind}")
        return
    sub.sort_values(["month", "share"], ascending=[True, False]).to_csv(out_csv, index=False)
    wrote(out_csv)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main

def main():
    if not os.path.exists(TRENDS):
        raise FileNotFoundError(
            f"Trends CSV not found: {TRENDS}\n"
            "Run 04_apply_models_streaming.py first."
        )

    df = pd.read_csv(TRENDS)
    df["month"] = df["month"].astype(str)

    # stance trends
    plot_kind(
        df, "stance",
        os.path.join(VIZ_DIR, "stance_trends.png"),
        "Stance trends over time (supervised)",
        plot_key="stance_trends"
    )

    # regime trends
    plot_kind(
        df, "regime",
        os.path.join(VIZ_DIR, "regime_trends.png"),
        "Narrative regime trends over time (supervised)",
        top_n=12,
        plot_key="regime_trends"
    )

    # stance × Regime trends (top 12 trajectories by total share)
    sx = df[df["kind"] == "stance_x_regime"].copy()
    if not sx.empty:
        totals = sx.groupby("label")["share"].sum().sort_values(ascending=False).head(12).index
        sx = sx[sx["label"].isin(totals)]
        pivot = sx.pivot(index="month", columns="label", values="share").fillna(0.0).sort_index()
        pivot = apply_regime_filter(pivot, "stance_x_regime_top12")

        plt.figure(figsize=(12, 6))
        for c in pivot.columns:
            plt.plot(pivot.index, pivot[c],
             label=display_name(c),
             color=label_color(c))
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Month")
        plt.ylabel("Share of monthly attention")
        plt.title("Top stance × regime trajectories (top 12)")
        plt.legend(loc="best", fontsize=7)
        savefig(os.path.join(VIZ_DIR, "stance_x_regime_top12.png"))
    else:
        print("Skip stance_x_regime plot: no stance_x_regime rows found.")

    # raw count charts and overlaid charts
    plot_kind_counts(
        df, "stance",
        os.path.join(VIZ_DIR, "stance_counts.png"),
        "Post counts per stance per month (unweighted)",
        plot_key="stance_counts"
    )
    plot_kind_counts(
        df, "regime",
        os.path.join(VIZ_DIR, "regime_counts.png"),
        "Post counts per narrative regime per month (unweighted)",
        top_n=12,
        plot_key="regime_counts"
    )
    plot_kind_overlaid(
        df, "stance",
        os.path.join(VIZ_DIR, "stance_overlaid.png"),
        "Stance — post counts (bars) vs. engagement-weighted share (lines)",
        plot_key="stance_overlaid"
    )
    plot_kind_overlaid(
        df, "regime",
        os.path.join(VIZ_DIR, "regime_overlaid.png"),
        "Regime — post counts (bars) vs. engagement-weighted share (lines)",
        top_n=12,
        plot_key="regime_overlaid"
    )


    # separate (non-overlaid) versions of the above charts
    plot_kind_separate(
        df, "stance",
        os.path.join(VIZ_DIR, "stance_counts_separate.png"),
        os.path.join(VIZ_DIR, "stance_shares_separate.png"),
        "Stance — post counts per month (unweighted)",
        "Stance — engagement-weighted share per month",
        plot_key="stance_separate"
    )
    plot_kind_separate(
        df, "regime",
        os.path.join(VIZ_DIR, "regime_counts_separate.png"),
        os.path.join(VIZ_DIR, "regime_shares_separate.png"),
        "Narrative regime — post counts per month (unweighted)",
        "Narrative regime — engagement-weighted share per month",
        top_n=12,
        plot_key="regime_separate"
    )

    # summary table: regime + stance shares and raw counts in one place
    export_summary_table(
        df,
        os.path.join(VIZ_DIR, "corpus_summary_table.csv"),
        os.path.join(VIZ_DIR, "corpus_summary_table.md"),
    )

    # per-regime stance composition
    export_per_regime_stance(
        df,
        os.path.join(VIZ_DIR, "per_regime_stance.csv"),
        os.path.join(VIZ_DIR, "per_regime_stance.md"),
    )

    # stance x regime segmented bar chart (raw post counts)
    plot_stance_regime_segmented(
        df,
        os.path.join(VIZ_DIR, "stance_regime_segmented_counts.png"),
        "Post volume by stance and regime per month (bars=stances, segments=regimes)",
        use_counts=True,
        plot_key="stance_regime_segmented"
    )

    # stance x regime segmented bar chart (engagement-weighted)
    plot_stance_regime_segmented(
        df,
        os.path.join(VIZ_DIR, "stance_regime_segmented_weighted.png"),
        "Engagement-weighted volume by stance and regime per month (bars=stances, segments=regimes)",
        use_counts=False,
        plot_key="stance_regime_segmented"
    )
    # spike detection (stance + regime)
    spike_table_for_kind(df, "stance", os.path.join(VIZ_DIR, "stance_spikes.csv"), z=SPIKE_Z)
    spike_table_for_kind(df, "regime", os.path.join(VIZ_DIR, "regime_spikes.csv"), z=SPIKE_Z)

    # raw counts (weight_sum)
    export_raw_counts(df, "stance", os.path.join(VIZ_DIR, "stance_raw_counts.csv"))
    export_raw_counts(df, "regime", os.path.join(VIZ_DIR, "regime_raw_counts.csv"))

    # temporal co-attention of regimes
    if os.path.exists(TOPJSONL):
        co = build_regime_coattention_from_topjsonl(TOPJSONL)
        if not co.empty:
            co_out = os.path.join(VIZ_DIR, "regime_temporal_coattention.csv")
            co.to_csv(co_out, index=False)
            wrote(co_out)
        else:
            print("Skip temporal co-attention: exemplar JSONL produced no rows.")
    else:
        print("Skip temporal co-attention: exemplar JSONL not found:", TOPJSONL)

    # corpus-level summary statistics + monthly volume chart + summary card
    author_stats = count_authors_from_jsonl(DATA_PATH)

    monthly_stats = export_corpus_statistics(
        df,
        os.path.join(VIZ_DIR, "corpus_monthly_statistics.csv"),
        os.path.join(VIZ_DIR, "corpus_summary.md"),
        author_stats=author_stats,
    )
    plot_corpus_volume(
        monthly_stats,
        os.path.join(VIZ_DIR, "corpus_monthly_volume.png"),
    )
    plot_corpus_volume_separate(
        monthly_stats,
        os.path.join(VIZ_DIR, "corpus_monthly_posts.png"),
        os.path.join(VIZ_DIR, "corpus_monthly_engagement.png"),
    )
    plot_corpus_summary_table(
        monthly_stats,
        author_stats,
        os.path.join(VIZ_DIR, "corpus_summary_card.png"),
    )

    # engagement amplification per stance class + bar chart
    stance_amp = export_amplification(
        df, "stance",
        os.path.join(VIZ_DIR, "stance_amplification.csv"),
        os.path.join(VIZ_DIR, "stance_amplification.md"),
    )
    plot_amplification(
        stance_amp,
        os.path.join(VIZ_DIR, "stance_amplification.png"),
        kind="stance",
    )

    # engagement amplification per regime + bar chart
    regime_amp = export_amplification(
        df, "regime",
        os.path.join(VIZ_DIR, "engagement_amplification.csv"),
        os.path.join(VIZ_DIR, "engagement_amplification.md"),
    )
    plot_amplification(
        regime_amp,
        os.path.join(VIZ_DIR, "engagement_amplification.png"),
        kind="regime",
    )

    # summary
    print(f"\nAll outputs written to: {VIZ_DIR}")
    print(f"Files written this run ({len(_written)}):")
    for p in _written:
        print(f"  {p}")

if __name__ == "__main__":
    main()