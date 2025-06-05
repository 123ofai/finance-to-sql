import re
import calendar
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score
from rapidfuzz import process, fuzz

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_CSV  = "../data/queries_with_period_id_and_period_view.csv"   # must include: NL_Query, period_id, period_view
OUTPUT_CSV = "../results/period_result_6.csv"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ─── LOAD DATA & MODEL ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
model = SentenceTransformer(MODEL_NAME)

# ─── KEYWORD LISTS FOR RULE‐BASED CHECK ─────────────────────────────────────
FTP_KEYWORDS = [
    r"\bfor the period\b", r"\bfor that period\b", r"\bjust that month\b",
    r"\bonly that quarter\b", r"\bas at\b", r"\bas at\s+(?:month|quarter)\b",
]
PRD_KEYWORDS = [
    r"\byear to date\b", r"\bytd\b", r"\bso far\b", r"\bcumulative\b",
    r"\bthrough\b", r"\bup to\b", r"\bas of\b", r"\bto date\b",
    r"\bsince the start of the year\b", r"\bmonth to date\b", r"\bmtd\b",
    r"\bquarter to date\b", r"\bqtd\b", r"\bthrough end of\b",
    r"\bthrough end-of-period\b"
]

# Regex patterns for period units (months, qtrs, halves, fiscal year)
months = [m.lower() for m in calendar.month_name if m]
month_regex = r"(?:{})".format("|".join(months))
quarter_regex = r"(?:q[1-4]|quarter\s*[1-4])"
half_regex = r"(?:h1|h2|first half|second half|half-year\s*[12])"
fy_regex = r"(?:fy\s*\d{2,4}|financial year)"

period_unit_regex = rf"{month_regex}|{quarter_regex}|{half_regex}|{fy_regex}"

def detect_view(nl: str) -> str:
    low = nl.lower()

    # 1) If any explicit PRD keyword, return PRD
    for pat in PRD_KEYWORDS:
        if re.search(pat, low):
            return "PRD"

    # 2) If "for <period-unit>" pattern without PRD keyword, return FTP
    if re.search(rf"\bfor\b.*\b{period_unit_regex}\b", low):
        return "FTP"

    # 3) Semantic fallback: match against verbose prototypes
    FTP_PROTOS_VERB = [
      "FTP can be defined as ‘for the period’ meaning only that month or quarter",
      "FTP can be defined as ‘for that period only’ meaning the single slice of time",
      "FTP can be defined as ‘at month end’ meaning only that month",
      "FTP can be defined as ‘year ended’ meaning the year‐end snapshot",
    ]
    PRD_PROTOS_VERB = [
      "PRD can be defined as ‘to date’ meaning cumulative up until now",
      "PRD can be defined as ‘year to date’ meaning aggregated so far this fiscal year",
      "PRD can be defined as ‘month to date’ meaning cumulative this month",
      "PRD can be defined as ‘quarter to date’ meaning cumulative this quarter",
      "PRD can be defined as ‘so far’ meaning sum of all periods up until date",
    ]
    VIEW_PROTOS = FTP_PROTOS_VERB + PRD_PROTOS_VERB
    view_embs = model.encode(VIEW_PROTOS, convert_to_tensor=True, normalize_embeddings=True)

    q_emb = model.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, view_embs)[0]
    best_idx   = int(sims.argmax().item())
    best_score = sims[best_idx].item()

    # 4) If best semantic score is low, default to FTP
    if best_score < 0.45:
        return "FTP"

    best_proto = VIEW_PROTOS[best_idx]
    return "FTP" if best_proto in FTP_PROTOS_VERB else "PRD"

# ─── MONTH & QUARTER & HALF CANDIDATES ─────────────────────────────────────
month_names_full = [m.lower() for m in calendar.month_name if m]
month_names_abbr = [m.lower()[:3] for m in calendar.month_name if m]
all_month_candidates = list(set(month_names_full + month_names_abbr))

quarter_candidates = ["q1", "q2", "q3", "q4",
                      "quarter 1", "quarter 2", "quarter 3", "quarter 4",
                      "1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]
half_candidates = ["h1", "h2", "first half", "second half", "half-year 1", "half-year 2"]

ORDINAL_MAP = {
    "first":1, "1st":1,
    "second":2, "2nd":2,
    "third":3, "3rd":3,
    "fourth":4, "4th":4
}
NUM_WORD_MAP = {
    "one":1, "two":2, "three":3, "four":4,
    "five":5, "six":6, "seven":7, "eight":8,
    "nine":9, "ten":10, "eleven":11, "twelve":12
}

def extract_year(nl: str) -> int:
    m = re.search(r"\b(19|20)\d{2}\b", nl)
    return int(m.group(0)) if m else datetime.now().year

def extract_nature(nl: str) -> str:
    low = nl.lower()
    if re.search(r"\bquarter\b|\bq[1-4]\b", low): return "FQ"
    if re.search(r"\bhalf\b|\bh1\b|\bh2\b",   low): return "FH"
    if re.search(r"\bfinancial year\b|\bfy\b", low): return "FY"
    return "M"

def fuzzy_match_token(token: str, candidates: list, threshold=75):
    # Return best match and its score
    match, score, _ = process.extractOne(token, candidates, scorer=fuzz.ratio)
    if score >= threshold:
        return match
    return None

def extract_sequence(nl: str, nature: str) -> int:
    low = nl.lower()

    # 1) EXPLICIT MONTH (skip index 0)
    if nature == "M":
        # Exact full name
        for i, name in enumerate(calendar.month_name[1:], start=1):
            if name.lower() in low:
                return i

        # Exact 3-letter abbreviation
        for i, name in enumerate(calendar.month_name[1:], start=1):
            if name.lower()[:3] in low:
                return i

        # Numeric word (“one”→1 … “twelve”→12)
        for word, val in NUM_WORD_MAP.items():
            if re.search(rf"\b{word}\b", low) and 1 <= val <= 12:
                return val

        # Regex "month <number>"
        m = re.search(r"month\s+(\d{1,2})", low)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 12:
                return num

        # Fuzzy match each token to month names/abbrs
        tokens = re.findall(r"\w+", low)
        for t in tokens:
            fm = fuzzy_match_token(t, all_month_candidates)
            if fm:
                if fm in month_names_full:
                    return month_names_full.index(fm) + 1
                if fm in month_names_abbr:
                    return month_names_abbr.index(fm) + 1

        # Relative "last month" / "last N months"
        if re.search(r"last\s+month", low):
            seq = datetime.now().month - 1
            return seq if seq >= 1 else 1

        m = re.search(r"last\s+(\d+)\s+months?", low)
        if m:
            n = int(m.group(1))
            seq = datetime.now().month - n
            return seq if seq >= 1 else 1

    # 2) EXPLICIT QUARTER  (unchanged)
    if nature == "FQ":
        m = re.search(r"q([1-4])", low)
        if m:
            return int(m.group(1))
        for w, n in ORDINAL_MAP.items():
            if f"{w} quarter" in low:
                return n
        tokens = re.findall(r"\w+", low)
        for t in tokens:
            fm = fuzzy_match_token(t, quarter_candidates)
            if fm:
                num = re.search(r"([1-4])", fm)
                if num:
                    return int(num.group(1))
        if re.search(r"last\s+quarter", low):
            anchor = (datetime.now().month - 1)//3 + 1
            seq = anchor - 1
            return seq if seq >= 1 else 1
        m = re.search(r"last\s+(\d+)\s+quarters?", low)
        if m:
            n = int(m.group(1))
            anchor = (datetime.now().month - 1)//3 + 1
            seq = anchor - n
            return seq if seq >= 1 else 1

    # 3) EXPLICIT HALF  (unchanged)
    if nature == "FH":
        if re.search(r"h1\b|first\s+half", low):
            return 1
        if re.search(r"h2\b|second\s+half", low):
            return 2
        for w, n in ORDINAL_MAP.items():
            if f"{w} half" in low:
                return n
        tokens = re.findall(r"\w+", low)
        for t in tokens:
            fm = fuzzy_match_token(t, half_candidates)
            if fm:
                if "1" in fm or "first" in fm:
                    return 1
                if "2" in fm or "second" in fm:
                    return 2
        month = datetime.now().month
        return 1 if month <= 6 else 2

    # 4) FISCAL YEAR
    if nature == "FY":
        return 1

    # 5) FALLBACK TO CURRENT PERIOD
    month = datetime.now().month
    if nature == "M":
        return month
    if nature == "FQ":
        return (month - 1)//3 + 1
    if nature == "FH":
        return 1 if month <= 6 else 2
    return 1

def construct_period_id(nl: str) -> str:
    year   = extract_year(nl)
    nature = extract_nature(nl)
    view   = detect_view(nl)
    seq    = extract_sequence(nl, nature)
    return f"{year}_{nature}_{view}_{seq}"

# ─── PARSE GROUND-TRUTH COMPONENTS ───────────────────────────────────────────
gt_split = df["period_id"].str.split("_", expand=True)
df["gt_year"]      = gt_split[0].astype(int)
df["gt_nature"]    = gt_split[1]
df["gt_view"]      = gt_split[2]
df["gt_sequence"]  = gt_split[3].astype(int)

# ─── PREDICT COMPONENTS ─────────────────────────────────────────────────────
df["predicted_view"]      = df["NL_Query"].apply(detect_view)
df["predicted_year"]      = df["NL_Query"].apply(extract_year)
df["predicted_nature"]    = df["NL_Query"].apply(extract_nature)
df["predicted_sequence"]  = df.apply(lambda r: extract_sequence(r["NL_Query"], r["predicted_nature"]), axis=1)
df["predicted_period_id"] = df["NL_Query"].apply(construct_period_id)

# ─── EVALUATE & PRINT METRICS ────────────────────────────────────────────────
print("=== Period_ID ===")
print(f"Accuracy: {accuracy_score(df['period_id'], df['predicted_period_id']):.2%}")
print(f"Macro F1:  {f1_score(df['period_id'], df['predicted_period_id'], average='macro'):.2f}\n")

print("=== Year ===")
print(f"Accuracy: {accuracy_score(df['gt_year'], df['predicted_year']):.2%}")
print(f"Macro F1:  {f1_score(df['gt_year'], df['predicted_year'], average='macro'):.2f}\n")

print("=== Nature ===")
print(f"Accuracy: {accuracy_score(df['gt_nature'], df['predicted_nature']):.2%}")
print(f"Macro F1:  {f1_score(df['gt_nature'], df['predicted_nature'], average='macro'):.2f}\n")

print("=== View ===")
print(f"Accuracy: {accuracy_score(df['gt_view'], df['predicted_view']):.2%}")
print(f"Macro F1:  {f1_score(df['gt_view'], df['predicted_view'], average='macro'):.2f}\n")

print("=== Sequence ===")
print(f"Accuracy: {accuracy_score(df['gt_sequence'], df['predicted_sequence']):.2%}")
print(f"Macro F1:  {f1_score(df['gt_sequence'], df['predicted_sequence'], average='macro'):.2f}\n")

# ─── SAVE RESULTS ──────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions and metrics saved to {OUTPUT_CSV}")
