import os
import json
import re
import calendar
from datetime import datetime
import torch
import tempfile
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import sqlalchemy

# ─── CONFIG ────────────────────────────────────────────────────────────
# Adjust paths relative to the repo root
#GLOSSARY_CSV        = os.path.join("/opt/ml/model", "data/1b_glossary_descriptions.csv")
#GROUPING_MASTER_CSV = os.path.join("/opt/ml/model", "data/fbi_grouping_master.csv")


SOURCE_DIR_MODELS = os.environ.get("SM_MODEL_DIR", "/opt/ml/model") + '/'
GLOSSARY_CSV        = SOURCE_DIR_MODELS + "data/1b_glossary_descriptions.csv"
GROUPING_MASTER_CSV = SOURCE_DIR_MODELS + "data/fbi_grouping_master.csv"

# Schema + table
SCHEMA      = "epm1-replica.finalyzer.info_100032"
TABLE       = "fbi_entity_analysis_report"

# Default report parameters
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"

# Hyperparameters for NL→Glossary & Glossary→Grouping
TOP_K             = 5
W_SIM1, W_RERANK1 = 0.5, 0.5  # Stage1
W_SIM2, W_RERANK2 = 0.6, 0.4  # Stage2

# ─── LOAD SECRETS ────────────────────────────────────────────────────────
# ─── LOAD SECRETS ───────────────────────────────────────────────────────
# Either from environment variables or load JSON files directly if paths are provided
ssh_env = os.environ.get("SSH_CONFIG_JSON")
if ssh_env:
    ssh_conf = json.loads(ssh_env)
else:
    with open(SOURCE_DIR_MODELS + "config/SSH_CONFIG.json", "r", encoding="utf-8") as f:
        ssh_conf = json.load(f)

pg_env = os.environ.get("PG_CONFIG_JSON")
if pg_env:
    pg_conf = json.loads(pg_env)
else:
    with open(SOURCE_DIR_MODELS + "config/PG_CONFIG.json", "r", encoding="utf-8") as f:
        pg_conf = json.load(f)

# ─── CACHE MODELS & DATA ─────────────────────────────────────────────────
def load_resources():
    # 1) Load data frames
    gloss_df = pd.read_csv(GLOSSARY_CSV)
    group_df = pd.read_csv(GROUPING_MASTER_CSV)

    # 2) Build enriched glossary texts
    def build_full_text(row):
        txt = f"{row['Glossary']} can be defined as {row['Description']}"
        if pd.notnull(row.get('Formulas, if any')):
            txt += f" Its Formula is: {row['Formulas, if any']}"
        return txt

    term_texts = gloss_df.apply(build_full_text, axis=1).tolist()
    label_texts = group_df['grouping_label'].tolist()

    # 3) Initialize models
    bi_encoder = SentenceTransformer(SOURCE_DIR_MODELS + 'models/bi_encoder')
    reranker_1 = CrossEncoder(SOURCE_DIR_MODELS + 'models/stage1_cross_encoder_finetuned_MiniLM_noisyhardnegative_v3_withdesc')
    reranker_2 = CrossEncoder(SOURCE_DIR_MODELS + 'models/stage2_cross_encoder_finetuned_MiniLM_hardnegative_v2')
    period_encoder = SentenceTransformer(SOURCE_DIR_MODELS + 'models/period_encoder')

    # 4) Precompute embeddings for Stage1 and Stage2
    with torch.no_grad():
        term_embs  = bi_encoder.encode(term_texts, convert_to_tensor=True, normalize_embeddings=True)
        label_embs = bi_encoder.encode(label_texts, convert_to_tensor=True, normalize_embeddings=True)
        view_embs  = period_encoder.encode(
            [
                "FTP can be defined as ‘for the period’ meaning only that month or quarter",
                "FTP can be defined as ‘for that period only’ meaning the single slice of time",
                "FTP can be defined as ‘at month end’ meaning only that month",
                "FTP can be defined as ‘year ended’ meaning the year‐end snapshot",
                "PRD can be defined as ‘to date’ meaning cumulative up until now",
                "PRD can be defined as ‘year to date’ meaning aggregated so far this fiscal year",
                "PRD can be defined as ‘month to date’ meaning cumulative this month",
                "PRD can be defined as ‘quarter to date’ meaning cumulative this quarter",
                "PRD can be defined as ‘so far’ meaning sum of all periods up until date"
            ],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    return (
        gloss_df, group_df,
        bi_encoder, reranker_1, reranker_2,
        term_texts, term_embs,
        label_texts, label_embs,
        period_encoder, view_embs
    )

# Loading all these resources
(
    gloss_df, group_df,
    bi_encoder, reranker_1, reranker_2,
    term_texts, term_embs,
    label_texts, label_embs,
    period_encoder, view_embs
) = load_resources()

# ─── PIPELINE FUNCTIONS: STAGE 1 & 2 ─────────────────────────────────────
def extract_glossary(nl_query: str) -> str:
    q_emb = bi_encoder.encode(nl_query, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, term_embs)[0]
    top_idx    = torch.topk(sims, k=TOP_K).indices.tolist()
    top_terms  = [term_texts[i] for i in top_idx]
    top_sims   = [sims[i].item() for i in top_idx]

    pairs         = [(nl_query, t) for t in top_terms]
    rerank_scores = reranker_1.predict(pairs)
    final_scores  = [W_SIM1*s + W_RERANK1*r for s, r in zip(top_sims, rerank_scores)]
    best          = int(torch.tensor(final_scores).argmax().item())
    return top_terms[best].split(' can be defined as ')[0]

def lookup_grouping(gloss_term: str) -> (str,int):
    q_emb = bi_encoder.encode(gloss_term, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, label_embs)[0]
    top_idx    = torch.topk(sims, k=TOP_K).indices.tolist()
    top_labels = [label_texts[i] for i in top_idx]
    top_sims   = [sims[i].item() for i in top_idx]

    pairs         = [(gloss_term, lbl) for lbl in top_labels]
    rerank_scores = reranker_2.predict(pairs)
    final_scores  = [W_SIM2*s + W_RERANK2*r for s, r in zip(top_sims, rerank_scores)]
    best          = int(torch.tensor(final_scores).argmax().item())
    lbl           = top_labels[best]
    gid           = int(group_df.loc[group_df['grouping_label']==lbl, 'grouping_id'].iat[0])
    return lbl, gid

# ─── PERIOD RESOLUTION FUNCTIONS ─────────────────────────────────────────
# Build regex for any period unit
months = [m.lower() for m in calendar.month_name if m]
month_regex   = r"(?:{})".format("|".join(months))
quarter_regex = r"(?:q[1-4]|quarter\s*[1-4])"
half_regex    = r"(?:h1|h2|first half|second half|half-year\s*[12])"
fy_regex      = r"(?:fy\s*\d{2,4}|financial year)"
period_unit_regex = rf"{month_regex}|{quarter_regex}|{half_regex}|{fy_regex}"

def detect_view(nl: str) -> str:
    # Keywords for view detection
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
    low = nl.lower()

    # 1) If any explicit PRD keyword, return PRD
    for pat in PRD_KEYWORDS:
        if re.search(pat, low):
            return "PRD"

    # 2) If "for <period-unit>" pattern without PRD keyword, return FTP
    if re.search(rf"\bfor\b.*\b{period_unit_regex}\b", low):
        return "FTP"

    # 3) Semantic fallback
    q_emb = period_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, view_embs)[0]
    best_idx   = int(sims.argmax().item())
    best_score = sims[best_idx].item()

    # 4) If best semantic score is low, default to FTP
    if best_score < 0.45:
        return "FTP"

    return "FTP" if best_idx < 4 else "PRD"


# Candidates for fuzzy matching
month_names_full = [m.lower() for m in calendar.month_name if m]
month_names_abbr = [m[:3].lower() for m in calendar.month_name if m]
all_month_candidates = list(set(month_names_full + month_names_abbr))

quarter_candidates = [
    "q1","q2","q3","q4",
    "quarter 1","quarter 2","quarter 3","quarter 4",
    "1st quarter","2nd quarter","3rd quarter","4th quarter"
]
half_candidates = ["h1","h2","first half","second half","half-year 1","half-year 2"]

ORDINAL_MAP = {
    "first":1,"1st":1,
    "second":2,"2nd":2,
    "third":3,"3rd":3,
    "fourth":4,"4th":4
}
NUM_WORD_MAP = {
    "one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,
    "nine":9,"ten":10,"eleven":11,"twelve":12
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
    match, score, _ = process.extractOne(token, candidates, scorer=fuzz.ratio)
    return match if score >= threshold else None

def extract_sequence(nl: str, nature: str) -> int:
    low = nl.lower()

    # 1) EXPLICIT MONTH
    if nature == "M":
        # (a) Exact full month
        for i, name in enumerate(calendar.month_name[1:], start=1):
            if name.lower() in low:
                return i
        # (b) Exact 3-letters
        for i, name in enumerate(calendar.month_name[1:], start=1):
            if name.lower()[:3] in low:
                return i
        # (c) Numeric word “one” → 1 … “twelve” → 12
        for word, val in NUM_WORD_MAP.items():
            if re.search(rf"\b{word}\b", low) and 1 <= val <= 12:
                return val
        # (d) Regex “month <number>”
        m = re.search(r"month\s+(\d{1,2})", low)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 12:
                return num
        # (e) Fuzzy-match tokens
        tokens = re.findall(r"\w+", low)
        for t in tokens:
            fm = fuzzy_match_token(t, all_month_candidates)
            if fm:
                if fm in month_names_full:
                    return month_names_full.index(fm) + 1
                if fm in month_names_abbr:
                    return month_names_abbr.index(fm) + 1
        # (f) “last month” or “last N months”
        if re.search(r"last\s+month", low):
            seq = datetime.now().month - 1
            return seq if seq >= 1 else 1
        m = re.search(r"last\s+(\d+)\s+months?", low)
        if m:
            n = int(m.group(1))
            seq = datetime.now().month - n
            return seq if seq >= 1 else 1

    # 2) EXPLICIT QUARTER
    if nature == "FQ":
        m = re.search(r"q([1-4])", low)
        if m:
            return int(m.group(1))
        for w, val in ORDINAL_MAP.items():
            if f"{w} quarter" in low:
                return val
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

    # 3) EXPLICIT HALF
    if nature == "FH":
        if re.search(r"h1\b|first\s+half", low):
            return 1
        if re.search(r"h2\b|second\s+half", low):
            return 2
        for w, val in ORDINAL_MAP.items():
            if f"{w} half" in low:
                return val
        tokens = re.findall(r"\w+", low)
        for t in tokens:
            fm = fuzzy_match_token(t, half_candidates)
            if fm:
                if "1" in fm or "first" in fm:
                    return 1
                if "2" in fm or "second" in fm:
                    return 2
        # Fallback half by current month
        month = datetime.now().month
        return 1 if month <= 6 else 2

    # 4) FISCAL YEAR → always 1
    if nature == "FY":
        return 1

    # 5) FALLBACK: current period
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


# ─── ENTRYPOINT ─────────────────────────────────────────────────────────
def handler(event, context=None):
    # Parse incoming payload
    try:
        payload = json.loads(event.get('body', json.dumps(event)))
    except json.JSONDecodeError:
        payload = event
    nl_query = payload.get('query', '')

    # 1) NL → Glossary
    gloss = extract_glossary(nl_query)

    # 2) Glossary → Grouping
    label, gid = lookup_grouping(gloss)

    # 3) Period, Scenario, Nature, Sequence
    period_id = construct_period_id(nl_query)

    # We can still infer scenario/nature as before
    scenario = 'Forecast' if 'forecast' in nl_query.lower() or 'budget' in nl_query.lower() else \
                ('Cashflow' if 'cash' in nl_query.lower() else 'Actual')
    nature   = 'Standalone'

    # Build SQL
    params = {
        'entity_id':    DEFAULT_ENTITY_ID,
        'grouping_id':  gid,
        'period_id':    period_id,
        'nature':       nature,
        'scenario':     scenario,
        'taxonomy':     DEFAULT_TAXONOMY,
        'currency':     DEFAULT_CURRENCY
    }
    sql = f"""
            SELECT value
            FROM "{SCHEMA}"."{TABLE}"
            WHERE entity_id={params['entity_id']}
            AND grouping_id={params['grouping_id']}
            AND period_id='{params['period_id']}'
            AND nature_of_report='{params['nature']}'
            AND scenario='{params['scenario']}'
            AND taxonomy_id={params['taxonomy']}
            AND reporting_currency='{params['currency']}';
            """

    # Write SSH key to a temp file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tf:
        tf.write(ssh_conf['ssh_pkey'])
        tf.flush()

        with SSHTunnelForwarder(
            (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
            ssh_username=ssh_conf['ssh_username'],
            ssh_pkey=tf.name,
            remote_bind_address=(pg_conf['host'], pg_conf['port'])
        ) as tunnel:
            local_port = tunnel.local_bind_port
            conn_str = (
                f"postgresql://{pg_conf['user']}:{pg_conf['password']}"
                f"@127.0.0.1:{local_port}/{pg_conf['dbname']}"
            )
            engine  = sqlalchemy.create_engine(conn_str)
            df = pd.read_sql(sql, engine)

    return {
        "statusCode": 200,
        "body": df.to_json(orient="records")
    }


#if __name__ == "__main__":
#    print(handler({"query": "profit in 2024"}))