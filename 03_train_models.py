
"""
Train the stance, regime, and gift-flag classifiers.

This script takes the labelled CSV and trains three classifiers over sentence-
transformer embeddings of the post text.

All three share the same embedding backbone (EMBED_MODEL_NAME in
config.py, currently paraphrase-multilingual-mpnet-base-v2). The classifier head
is logistic regression in every case, with class_weight="balanced"
to compensate for the imbalanced label distribution observed in the
labelled set.


Notes on the gift-flag model:
Though the gift/reciprocity flag is part of the labelling schema and the model is trained here,
it is not used in the thesis analysis.


Notes on the threshold-tuning evaluation:
Per-class regime thresholds are selected on the validation set by
maximising F1 over a 0.20-0.80 grid in 0.05 steps. The classification
report immediately below the tuning step is computed on the same
validation set, so the reported per-class F1 figures are optimistic
relative to a held-out test set. The split is therefore reported as
threshold-tuning-and-validation rather than as an independent test
estimate. 
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sentence_transformers import SentenceTransformer
from config import WORK_DIR, STANCE_LABELS, REGIME_LABELS, EMBED_MODEL_NAME, RANDOM_SEED

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Output paths

LABELS_CSV   = os.path.join(WORK_DIR, "labels_sample.csv")
ENCODERS_OUT = os.path.join(WORK_DIR, "encoders.joblib")

_stamp           = datetime.now().strftime("%Y%m%d_%H%M")
STANCE_MODEL_OUT = os.path.join(WORK_DIR, f"stance_model_{_stamp}.joblib")
REGIME_MODEL_OUT = os.path.join(WORK_DIR, f"regime_model_{_stamp}.joblib")
GIFT_MODEL_OUT   = os.path.join(WORK_DIR, f"gift_model_{_stamp}.joblib")
REPORTS_OUT      = os.path.join(WORK_DIR, f"classification_reports_{_stamp}.txt")
METRICS_JSON_OUT = os.path.join(WORK_DIR, f"training_metrics_{_stamp}.json")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Text and label preprocessing helpers

def strip_alt_markers(text):
 # remove [ALT] provenance markers from the labelling CSV before training   
    return re.sub(r"\[ALT\]\s*", "", str(text)).strip()

def parse_regimes(s):
    if not isinstance(s, str) or not s.strip():
        return []
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return [p for p in parts if p in REGIME_LABELS]

def stratify_proxy_multilabel(Y):
    label_count = Y.sum(axis=1).astype(int)
    top_label   = np.where(Y.sum(axis=1) > 0, Y.argmax(axis=1), -1)
    return np.array([f"{lc}_{tl}" for lc, tl in zip(label_count, top_label)])

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main

def main():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(LABELS_CSV)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # load and clean the labelled CSV
    df = pd.read_csv(LABELS_CSV)

    for col in ["stance", "regimes", "gift_reciprocity_flag", "text"]:
        if col not in df.columns:
            df[col] = ""
    df[["stance", "regimes", "gift_reciprocity_flag", "text"]] = df[
        ["stance", "regimes", "gift_reciprocity_flag", "text"]
    ].fillna("")

    # strip the [ALT] markers before any model touches the text
    df["text"] = df["text"].apply(strip_alt_markers)

    # filter to rows with a usable stance value
    n_before = len(df)
    df = df[df["stance"].astype(str).str.strip() != ""].copy()
    n_after_empty = len(df)
    df = df[df["stance"].isin(STANCE_LABELS)].copy()
    n_after_schema = len(df)

    if n_before != n_after_schema:
        dropped_empty = n_before - n_after_empty
        dropped_schema = n_after_empty - n_after_schema
        log(f"Stance filtering: dropped {dropped_empty} empty + {dropped_schema} out-of-schema "
            f"= {n_before - n_after_schema} of {n_before} rows.")

    # parse regime strings into label lists.
    df["reg_list"] = df["regimes"].apply(parse_regimes)
    df_reg = df[df["reg_list"].map(len) > 0].copy()

    log(f"Training run: {_stamp}")
    log(f"Labeled rows for stance:  {len(df)}")
    log(f"Labeled rows for regimes: {len(df_reg)}")

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Per-class label counts (full labelled set, before train/val split)
    # mlb_check is only for reporting the regime label counts
    mlb_check = MultiLabelBinarizer(classes=REGIME_LABELS)
    Y_check   = mlb_check.fit_transform(df_reg["reg_list"])
    log("\nRegime label counts (all labeled rows):")
    for i, label in enumerate(mlb_check.classes_):
        log(f"  {label}: {int(Y_check[:, i].sum())}")

    log("\nStance label counts:")
    for s in STANCE_LABELS:
        log(f"  {s}: {int((df['stance'] == s).sum())}")

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Embedding model
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    log(f"\nEmbedding model: {EMBED_MODEL_NAME}")

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # STANCE MODEL
    X_stance = embedder.encode(
        df["text"].tolist(),
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    stance_le = LabelEncoder()
    y_stance  = stance_le.fit_transform(df["stance"].astype(str))

    # 80/20 split, stratified by stance, seeded for reproducibility.
    Xtr, Xte, ytr, yte = train_test_split(
        X_stance, y_stance, test_size=0.2, random_state=RANDOM_SEED, stratify=y_stance
    )

    stance_clf = LogisticRegression(max_iter=3000, n_jobs=-1, class_weight="balanced")
    stance_clf.fit(Xtr, ytr)
    pred = stance_clf.predict(Xte)

    log("\n================ STANCE RESULTS ================")
    log(classification_report(yte, pred, target_names=stance_le.classes_, digits=3))
    log("Confusion matrix:")
    log(str(confusion_matrix(yte, pred)))

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # REGIME MODEL
    X_reg = embedder.encode(
        df_reg["text"].tolist(),
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    mlb   = MultiLabelBinarizer(classes=REGIME_LABELS)
    Y_reg = mlb.fit_transform(df_reg["reg_list"])

    strat_proxy = stratify_proxy_multilabel(Y_reg)
    try:
        Xtr2, Xte2, Ytr2, Yte2 = train_test_split(
            X_reg, Y_reg, test_size=0.2, random_state=RANDOM_SEED, stratify=strat_proxy
        )
    except ValueError:
        log("\nWARNING: Stratified split failed (minority class too small). "
            "Falling back to random split. Add more labeled examples.")
        Xtr2, Xte2, Ytr2, Yte2 = train_test_split(
            X_reg, Y_reg, test_size=0.2, random_state=RANDOM_SEED
        )

    base    = LogisticRegression(max_iter=3000, n_jobs=-1, class_weight="balanced")
    reg_clf = OneVsRestClassifier(base)
    reg_clf.fit(Xtr2, Ytr2)

    proba = reg_clf.predict_proba(Xte2)

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Per-class threshold tuning

    log("\nTuning per-class thresholds on validation set...")
    best_thresholds  = []
    threshold_curves = {}
    for i, label in enumerate(mlb.classes_):
        best_f1 = 0.0
        best_t  = 0.5
        curve   = []
        for t in np.arange(0.2, 0.81, 0.05):
            pred_t = (proba[:, i] >= t).astype(int)
            if pred_t.sum() == 0:
                curve.append({"threshold": round(float(t), 2), "f1": 0.0})
                continue
            f = f1_score(Yte2[:, i], pred_t, zero_division=0)
            curve.append({"threshold": round(float(t), 2), "f1": round(float(f), 4)})
            if f > best_f1:
                best_f1 = f
                best_t  = round(float(t), 2)
        best_thresholds.append(best_t)
        threshold_curves[label] = curve
        log(f"  {label}: threshold={best_t:.2f}, val F1={best_f1:.3f}")

    Ypred = np.column_stack([
        (proba[:, i] >= best_thresholds[i]).astype(int)
        for i in range(len(mlb.classes_))
    ])

    log("\n================ REGIME RESULTS (multi-label, tuned thresholds) ================")
    log(classification_report(Yte2, Ypred, target_names=mlb.classes_, digits=3))

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # GIFT MODEL
    df_gift = df.copy()

    df_gift["gift_y"] = (
        df_gift["gift_reciprocity_flag"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    df_gift = df_gift[df_gift["gift_y"].isin(["0", "1"])].copy()

    log(f"\nLabeled rows for gift flag: {len(df_gift)}")
    log(f"  (gift value counts: {df_gift['gift_y'].value_counts().to_dict()})")

    gift_saved = False
    if len(df_gift) >= 10:
        X_gift = embedder.encode(
            df_gift["text"].tolist(),
            batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        y_gift = df_gift["gift_y"].astype(int).values

        can_stratify = (len(np.unique(y_gift)) == 2 and min(np.bincount(y_gift)) >= 2)
        Xtr3, Xte3, ytr3, yte3 = train_test_split(
            X_gift, y_gift, test_size=0.2, random_state=RANDOM_SEED,
            stratify=y_gift if can_stratify else None
        )

        gift_clf = LogisticRegression(max_iter=3000, n_jobs=-1, class_weight="balanced")
        gift_clf.fit(Xtr3, ytr3)
        pred3 = gift_clf.predict(Xte3)

        log("\n================ GIFT FLAG RESULTS (binary) ================")
        log(classification_report(yte3, pred3, digits=3))

        dump({"model": gift_clf, "embed_model": EMBED_MODEL_NAME}, GIFT_MODEL_OUT)
        log(f"Saved: {GIFT_MODEL_OUT}")
        gift_saved = True
    else:
        log("Skipping gift model training (need at least 10 labeled rows with 0/1).")

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # SAVE MODELS AND ENCODERS

    dump({"model": stance_clf, "embed_model": EMBED_MODEL_NAME}, STANCE_MODEL_OUT)
    dump({
        "model":      reg_clf,
        "embed_model": EMBED_MODEL_NAME,
        "thresholds": best_thresholds
    }, REGIME_MODEL_OUT)
    dump({"stance_le": stance_le, "mlb": mlb}, ENCODERS_OUT)

    log("\nSaved:")
    log(f"  {STANCE_MODEL_OUT}")
    log(f"  {REGIME_MODEL_OUT}")
    log(f"  {ENCODERS_OUT}")
    if gift_saved:
        log(f"  {GIFT_MODEL_OUT}")
    else:
        log("  gift model NOT saved (insufficient gift labels)")

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # SAVE STRUCTURED METRICS JSON

    stance_p, stance_r, stance_f, stance_sup = precision_recall_fscore_support(
        yte, pred,
        labels=list(range(len(stance_le.classes_))),
        zero_division=0
    )
    stance_cm = confusion_matrix(yte, pred).tolist()

    stance_metrics = {
        "classes":          list(stance_le.classes_),
        "confusion_matrix": stance_cm,
        "per_class": {
            stance_le.classes_[i]: {
                "precision": round(float(stance_p[i]), 4),
                "recall":    round(float(stance_r[i]), 4),
                "f1":        round(float(stance_f[i]), 4),
                "support":   int(stance_sup[i]),
            }
            for i in range(len(stance_le.classes_))
        }
    }

    # Regime metrics: per-class numbers plus the chosen thresholds and full grid of threshold/F1 pairs
    regime_p, regime_r, regime_f, regime_sup = precision_recall_fscore_support(
        Yte2, Ypred, zero_division=0
    )
    regime_metrics = {
        "classes":          list(mlb.classes_),
        "thresholds":       {mlb.classes_[i]: best_thresholds[i]
                             for i in range(len(mlb.classes_))},
        "threshold_curves": threshold_curves,
        "per_class": {
            mlb.classes_[i]: {
                "precision": round(float(regime_p[i]), 4),
                "recall":    round(float(regime_r[i]), 4),
                "f1":        round(float(regime_f[i]), 4),
                "support":   int(regime_sup[i]),
                "threshold": best_thresholds[i],
            }
            for i in range(len(mlb.classes_))
        }
    }

    # class distributions, full labelled set
    stance_dist = {s: int((df["stance"] == s).sum()) for s in STANCE_LABELS}
    regime_dist = {mlb_check.classes_[i]: int(Y_check[:, i].sum())
                   for i in range(len(mlb_check.classes_))}

    metrics_out = {
        "stamp":          _stamp,
        "embed_model":    EMBED_MODEL_NAME,
        "n_stance_train": len(df),
        "n_regime_train": len(df_reg),
        "stance_dist":    stance_dist,
        "regime_dist":    regime_dist,
        "stance":         stance_metrics,
        "regime":         regime_metrics,
    }

    with open(METRICS_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    log(f"\nMetrics JSON saved to: {METRICS_JSON_OUT}")

    # save text report
    with open(REPORTS_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Reports saved to: {REPORTS_OUT}")

if __name__ == "__main__":
    main()