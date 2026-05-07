
"""
Streamlit UI for manual sample labelling.

Reads the labelling sample produced by 01_make_label_sample.py, presents one
post at a time alongside the regime-tagging guidelines, and writes the
labeller's choices back into the same CSV. Each save updates four
columns: stance, regimes, gift_reciprocity_flag, and notes.

"""

import os
import pandas as pd
import streamlit as st
from config import WORK_DIR, STANCE_LABELS, REGIME_LABELS

CSV_PATH = os.path.join(WORK_DIR, "labels_sample.csv")

st.set_page_config(layout="wide")
st.title("Labeling UI — Stance + Narrative Regimes (+ Gift/Reciprocity flag)")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Read CSV and normalise label columns

if not os.path.exists(CSV_PATH):
    st.error(f"Missing {CSV_PATH}. Run 01_make_label_sample.py first.")
    st.stop()

df = pd.read_csv(CSV_PATH).reset_index(drop=True)

# ensure all label columns exist
for col in ["stance", "regimes", "gift_reciprocity_flag", "notes"]:
    if col not in df.columns:
        df[col] = ""

# replace NaN with empty string in label columns so that downstream string operations behave consistently
df[["stance", "regimes", "gift_reciprocity_flag", "notes"]] = df[
    ["stance", "regimes", "gift_reciprocity_flag", "notes"]
].fillna("")

if df.empty:
    st.error("CSV has 0 rows.")
    st.stop()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Check if rows are labelled or unlabelled

unlabeled = df[
    (df["stance"].astype(str).str.strip() == "") |
    (df["regimes"].astype(str).str.strip() == "") |
    (df["gift_reciprocity_flag"].astype(str).str.strip() == "")
]

st.sidebar.write(f"Total rows: {len(df)}")
st.sidebar.write(f"Unlabeled rows: {len(unlabeled)}")

default_idx = int(unlabeled.index[0]) if len(unlabeled) else 0

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Row navigation via session state

if "row_index" not in st.session_state:
    st.session_state["row_index"] = default_idx

idx = st.sidebar.number_input(
    "Row index",
    min_value=0,
    max_value=len(df) - 1,
    value=int(st.session_state["row_index"])
)
idx = int(idx)
st.session_state["row_index"] = idx

row = df.loc[idx]
text = str(row.get("text", ""))
month = str(row.get("month", ""))
weight = row.get("weight", "")
sid = row.get("sample_id", "")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Sync the current row's values into session state

raw_regimes = str(row.get("regimes", "") or "")
existing_regimes = [r for r in raw_regimes.split("|") if r and r in REGIME_LABELS]

row_stance = str(row.get("stance", "") or "").strip()
existing_notes = str(row.get("notes", "") or "")

if st.session_state.get("last_idx") != idx:
    st.session_state["last_idx"] = idx
    st.session_state["regimes_multiselect"] = existing_regimes
    st.session_state["notes_textarea"] = existing_notes

col1, col2 = st.columns([3, 2])

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Left column: post text

with col1:
    st.subheader("Post text")
    st.write(text)
    st.caption(f"month={month} | weight={weight} | id={sid}")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Regime tagging guidelines (collapsible reference panel)

with st.expander("Regime Tagging Guidelines (Click to Expand)", expanded=False):
    st.markdown("""
### How to Tag Narrative Regimes

**Narrative logics**, not sentiment.

A post can have multiple regimes (usually 1–2).

---

#### 🟥 Extraction_Dispossession
- Core Narrative Logic
AI art is framed as an extractive system that appropriates creative labor without consent, compensation, or reciprocity.

AI framed as theft, plunder, exploitation, enclosure, capitalism, environmental extraction, industrial-scale scraping.
Centers around: labor value, ownership, consent, capital accumulation, platform capitalism, environmental cost.
Keywords: stealing, stolen, plagiarism, exploitation, corporate greed.

---

#### 🟦 Human_Essence_Ontology
- Core Narrative Logic
Art is inherently human. AI lacks intentionality, consciousness, embodiment, or interiority.

Claims that only humans make art; emphasis on soul, embodiment, intention, skill.
Centers around: embodiment, craft, intention, spirit, soul, human creativity (ontological), lived experience.
Keywords: real art is human, creativity, human spirit.

---

#### 🟨 Aesthetic_Pollution_Epistemic_Corruption
- Core Narrative Logic
AI degrades aesthetic culture and destabilizes epistemic trust.

AI described as slop, pollution, contamination, cultural pollution, ruining aesthetic standards, degraded search (can't differentiate AI from real images), trust collapse.
Centers around: visual trust collapse, authenticity crisis, search degradation.
Keywords: slop, garbage, ruined search, can't tell what's real, contamination.

---

#### 🟩 Governance_Boundary_Policing
- Core Narrative Logic
Communities and institutions must regulate AI to protect norms.

Rules, bans, moderation, boundary work, rule enforcement, "No AI allowed", institutional enforcement.
Centers around: policy, bans, blocklists, disclosure rules, institutional statements.
Keywords: ban, block, policy, rule, tagging requirement.

---

#### 🟪 Ideology_Hype_Discourse_Wars
- Core Narrative Logic
AI is debated (metadiscourse) as a sociotechnical narrative: democratization, inevitability, progress, assistive technology.

Centers around: ML vs GenAI definitional debates, assistive-tech framing, anti-hype or anti-futurist critique, "genie out of the bottle" rhetoric, democratization claims.
Debates about AI narratives: democratization, inevitability, definitions.
Keywords: democratizes art, hype, ML vs GenAI, assistive tech debate.

---

#### 🟧 AI_Native_Subculture_Legitimation
- Core Narrative Logic
AI art is normalized as a creative medium and community identity.

Positive AI art identity, prompting as a craft, #AIArtist community practice as legitimate creators.
Centers around: #AiArtist identity, prompt sharing, workflow experimentation, positive celebration of AI aesthetics
Keywords: (hashtags AIArt, models, etc.), (prompt sharing)

---
                
#### 🟦 Human_Artist_Community_Reproduction
- Core Narrative Logic
Human-made art is circulated, celebrated, and reinforced as a community practice.

Posts that invite artists to share work, boost each other, participate in aesthetic prompts, or reaffirm human creative solidarity.
Centers around: art-sharing threads, "drop your art" prompts, mutual boosting, artist visibility campaigns, colective creative participation
Keywords: share your art, artists drop, boost artists, mutuals, art thread, show your work
      
---

#### 🟫 Adult_Content_NSFWAIGen
- Core Narrative Logic
AI is used for scalable erotic image production within platform economies. Other adult related content.

AI-generated NSFW mass production, promotional erotic AI circulation, automation of desire, monetized erotic economies
Centers around: NSFW AI production, high-volume posting, promotional erotic tagging, AI-specific porn identities
Keywords: (related hashtags)

---

### Quick Decision Guide

• About theft/exploitation? → Extraction  
• About what art *is*? → Human_Essence  
• About slop/pollution/trust? → Pollution  
• About rules/bans? → Governance  
• About narrative battles/definitions? → Ideology  
• AI art identity practice? → AI_Native  
• Art share? → Artist Community
• AI NSFW promo economy? → Adult_Content  
""")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Right column: labelling controls

with col2:
    # stance (single label).
    stance = st.selectbox(
        "Stance (single label)",
        STANCE_LABELS,
        index=STANCE_LABELS.index(row_stance) if row_stance in STANCE_LABELS else 0
    )

    # regimes (multi-select)
    regimes = st.multiselect(
        "Narrative regimes (usually 1–2)",
        REGIME_LABELS,
        key="regimes_multiselect"
    )

    # Gift/reciprocity flag, stored as 1/0
    existing_flag = str(row.get("gift_reciprocity_flag", "")).strip()
    default_flag = True if existing_flag == "1" else False
    gift_flag = st.checkbox("Gift / reciprocity present?", value=default_flag)

    # Free-text notes
    notes = st.text_area("Notes (optional)", key="notes_textarea")

    # Save
    if st.button("Save"):
        df.at[idx, "stance"] = stance
        df.at[idx, "regimes"] = "|".join(regimes)
        df.at[idx, "gift_reciprocity_flag"] = "1" if gift_flag else "0"
        df.at[idx, "notes"] = notes
        df.to_csv(CSV_PATH, index=False)
        st.success("Saved.")

    # Jump to the next unlabelled
    if st.button("Next unlabeled"):
        if len(unlabeled):
            next_candidates = unlabeled.index[unlabeled.index > idx]
            next_idx = int(next_candidates[0]) if len(next_candidates) else int(unlabeled.index[0])
            st.session_state["row_index"] = next_idx
            st.rerun()
        else:
            st.info("No unlabeled rows left 🎉")

# To run this app from the command line: py -m streamlit run 02_label_app.py