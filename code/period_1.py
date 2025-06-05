import re
import calendar
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_CSV = "../data/queries_with_period_id_and_period_view.csv"    # must have NL_Query, Metric, period_id, period_view
MODEL_NAME = "BAAI/bge-large-en-v1.5"
VIEW_PROTOS = ["for the period", "to date"]
OUTPUT_CSV = "../data/period_result_1.csv"

# ─── LOAD DATA & MODEL ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
model = SentenceTransformer(MODEL_NAME)

# embed view prototypes once
view_embs = model.encode(VIEW_PROTOS, convert_to_tensor=True, normalize_embeddings=True)

# ─── SEMANTIC VIEW DETECTION ────────────────────────────────────────────────
def detect_view(nl: str) -> str:
    q_emb = model.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, view_embs)[0]
    return "FTP" if sims[0] >= sims[1] else "PRD"

# ─── RULE‐BASED PARSERS ────────────────────────────────────────────────────
ORDINAL_MAP = {
    "first":1,"1st":1,"second":2,"2nd":2,
    "third":3,"3rd":3,"fourth":4,"4th":4
}

def extract_year(nl: str) -> int:
    m = re.search(r"\b(20\d{2})\b", nl)
    return int(m.group(1)) if m else datetime.now().year

def extract_nature(nl: str) -> str:
    lower = nl.lower()
    if re.search(r"\bquarter\b|\bq[1-4]\b", lower): return "FQ"
    if re.search(r"\bhalf\b|\bh1\b|\bh2\b",   lower): return "FH"
    if re.search(r"\bfinancial year\b|\bfy\b", lower): return "FY"
    return "M"

def extract_sequence(nl: str, nature: str) -> int:
    lower = nl.lower()
    # Month
    if nature == "M":
        for i,name in enumerate(calendar.month_name):
            if name.lower() in lower:
                return i
        m = re.search(r"month\s+(\d{1,2})", lower)
        if m: return int(m.group(1))
    # Quarter
    if nature == "FQ":
        m = re.search(r"q([1-4])", lower)
        if m: return int(m.group(1))
        for w,n in ORDINAL_MAP.items():
            if f"{w} quarter" in lower:
                return n
    # Half-year
    if nature == "FH":
        if "h1" in lower or "first half" in lower:  return 1
        if "h2" in lower or "second half" in lower: return 2
        for w,n in ORDINAL_MAP.items():
            if f"{w} half" in lower:
                return n
    # Fiscal year
    if nature == "FY":
        return 1
    # Fallback to anchor = current date
    month = datetime.now().month
    if nature == "M":  return month
    if nature == "FQ": return (month-1)//3 + 1
    if nature == "FH": return 1 if month <= 6 else 2
    return 1

def construct_period_id(nl: str) -> str:
    year   = extract_year(nl)
    nature = extract_nature(nl)
    view   = detect_view(nl)
    seq    = extract_sequence(nl, nature)
    return f"{year}_{nature}_{view}_{seq}"

# ─── PREDICT ────────────────────────────────────────────────────────────────
df["predicted_view"]      = df["NL_Query"].apply(detect_view)
df["predicted_period_id"] = df["NL_Query"].apply(construct_period_id)

# ─── EVALUATE & PRINT METRICS ───────────────────────────────────────────────
view_acc = accuracy_score(df["period_view"], df["predicted_view"])
view_f1  = f1_score(df["period_view"], df["predicted_view"], average="macro")

pid_acc  = accuracy_score(df["period_id"], df["predicted_period_id"])
pid_f1   = f1_score(df["period_id"], df["predicted_period_id"], average="macro")

print("\n=== View Prediction Metrics ===")
print(f"Accuracy: {view_acc:.2%}")
print(f"Macro F1:  {view_f1:.2f}")

print("\n=== Period_ID Prediction Metrics ===")
print(f"Accuracy: {pid_acc:.2%}")
print(f"Macro F1:  {pid_f1:.2f}")


# ─── SAVE RESULTS ───────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved predictions to: {OUTPUT_CSV}")