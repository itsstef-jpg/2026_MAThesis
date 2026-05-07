
import os
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Filesystem paths (loaded from .env)

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
WORK_DIR  = os.getenv("WORK_DIR")

_missing = [name for name, val in (("DATA_PATH", DATA_PATH), ("WORK_DIR", WORK_DIR)) if not val]
if _missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        f"Create a .env file in the project root with these keys, or copy "
        f".env.example and fill in the values."
    )

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Label schema
# stance and narrative regime (category) labels. Stance is single-label, regime is multi-label. 

STANCE_LABELS = ["pro", "anti", "mixed", "neutral"]

REGIME_LABELS = [
    "Extraction_Dispossession",
    "Human_Essence_Ontology",
    "Aesthetic_Pollution_Epistemic_Corruption",
    "Governance_Boundary_Policing",
    "Ideology_Hype_Discourse_Wars",
    "AI_Native_Subculture_Legitimation",
    "Human_Artist_Community_Reproduction",
    "Adult_Content_NSFWAIGen",
    "Other_Unclear",
]

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Model settings

EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Sampling settings

LABEL_SAMPLE_N = 2500       # 1500-4000 recommended
TOP_ENGAGEMENT_FRAC = 0.35

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Apply settings

APPLY_BATCH_SIZE = 2000
TOP_K_EXEMPLARS = 25

# per-class confidence thresholds
STANCE_MIN_CONF = 0.45
REGIME_MIN_CONF = 0.0

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Analysis settings

SPIKE_Z = 1.5

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Gift/reciprocity flag (legacy)

GIFT_MIN_CONF = 0.0 
GIFT_COL = "gift_reciprocity_flag"

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Corpus properties

TOTAL_CORPUS_POSTS = 1506228