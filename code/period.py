import re
import calendar
from datetime import datetime
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ─── 1. LOAD & PREPARE ────────────────────────────────────────────────────────

# Path to your test CSV
INPUT_CSV  = "../data/queries_with_period_id_and_period_view.csv"
OUTPUT_CSV = "../results/queries_with_predicted_period_and_view.csv"

# Load queries
df = pd.read_csv(INPUT_CSV)

# Load period master for anchor logic if needed
pm_df = pd.read_csv("../data/fbi_period_master.csv")

# Load a single bi-encoder for both view prototypes and optional future use
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Embed the two view prototypes
VIEW_PROTOS = ["for the period", "to date"]
with torch.no_grad():
    view_embs = model.encode(VIEW_PROTOS, convert_to_tensor=True, normalize_embeddings=True)

# YTD trigger patterns (for fallback if needed)
YTD_TRIGGERS = [
    r"\bytd\b", r"year to date", r"year-to-date", r"so far",
    r"up to date", r"up to\b", r"as of\b", r"cumulative", r"through"
]


# ─── 2. SEMANTIC VIEW DETECTOR ────────────────────────────────────────────────

def detect_view_semantic(nl: str) -> str:
    """
    Returns 'FTP' if the query is closer to 'for the period',
    or 'PRD' if it is closer to 'to date', via cosine sim.
    """
    q_emb = model.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, view_embs)[0]  # [sim_FTP, sim_PRD]
    return "FTP" if sims[0] >= sims[1] else "PRD"


# ─── 3. RULE‐BASED YEAR / NATURE / SEQUENCE ───────────────────────────────────

ORDINAL_MAP = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2,
    "third": 3, "3rd": 3, "fourth": 4, "4th": 4
}

def extract_year(nl: str) -> int:
    m = re.search(r"\b(20\d{2})\b", nl)
    return int(m.group(1)) if m else datetime.now().year

def extract_nature(nl: str) -> str:
    lower = nl.lower()
    if re.search(r"\bquarter\b|\bq[1-4]\b", lower): return "FQ"
    if re.search(r"\bhalf\b|\bh1\b|\bh2\b", lower):    return "FH"
    if re.search(r"\bfinancial year\b|\bfy\b", lower): return "FY"
    return "M"

def extract_sequence(nl: str, nature: str) -> int:
    lower = nl.lower()
    # Month names
    if nature == "M":
        for i,name in enumerate(calendar.month_name):
            if name.lower() in lower:
                return i
        m = re.search(r"month\s+(\d{1,2})", lower)
        if m: return int(m.group(1))
    # Quarters
    if nature == "FQ":
        m = re.search(r"q([1-4])", lower)
        if m: return int(m.group(1))
        for w,n in ORDINAL_MAP.items():
            if f"{w} quarter" in lower: return n
    # Halves
    if nature == "FH":
        if "h1" in lower or "first half" in lower:  return 1
        if "h2" in lower or "second half" in lower: return 2
        for w,n in ORDINAL_MAP.items():
            if f"{w} half" in lower: return n
    # FY
    if nature == "FY":
        return 1
    # Fallback to anchor
    month = datetime.now().month
    if nature == "M":  return month
    if nature == "FQ": return (month-1)//3 + 1
    if nature == "FH": return 1 if month<=6 else 2
    return 1

def construct_period_id(nl: str) -> str:
    year   = extract_year(nl)
    nature = extract_nature(nl)
    view   = detect_view_semantic(nl)
    seq    = extract_sequence(nl, nature)
    return f"{year}_{nature}_{view}_{seq}"


# ─── 4. APPLY TO DATAFRAME & SAVE ─────────────────────────────────────────────

df["predicted_view"]      = df["NL_Query"].apply(detect_view_semantic)
df["predicted_period_id"] = df["NL_Query"].apply(construct_period_id)

# Reorder or rename columns as you like
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")
