import os
import json
import re
import ast
import operator as op
import tempfile
import calendar
from datetime import datetime

import torch
import logging
import pandas as pd
import sqlalchemy
from sshtunnel import SSHTunnelForwarder
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from rapidfuzz import process, fuzz


# ─── LOGGER CONFIGURATION ───────────────────────────────────────────────────
logger = logging.getLogger("inference")  
logger.setLevel(logging.INFO)       

# ─── CONFIG & CONSTANTS ─────────────────────────────────────────────────────
SCHEMA            = "epm1-replica.finalyzer.info_100032"
TABLE             = "fbi_entity_analysis_report"
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"
TOP_K             = 5
W_SIM1, W_RERANK1 = 0.5, 0.5
W_SIM2, W_RERANK2 = 0.6, 0.4

# ─── SECRETS FROM ENV ────────────────────────────────────────────────────────
ssh_conf   = {
    "tunnel_host": "3.110.31.32",
    "tunnel_port": 22,
    "ssh_username": "ec2-user",
    "ssh_pkey": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpQIBAAKCAQEAue87byNbSPgjucopUb6GoGqWRwjhgeSXOQsCKvo/o4FlVK83\nVsJAGMms/r20lT0MK+u8B3CJ/QNshJEd4lkshDw/Ei4Eh9zpMHw37wecPiIO0b/w\nbtvfVALp1ww/sEjjGib7Wv1BK9dlwVuHWzVHClSkfu/a52UsVwqmSnXKTCY0+dn3\njeDYA3Yyuo8ngp7V7588DH7IP2M2gePzO7zjOfFzPWlHgmsxEckzUnvDIiODYiLl\nnT9Rul3LhwsMmbUEV9Qy6671u+NuKzQaxbKmK6Nhsp5W12xsVZWjjsLBP78GW/43\nN1JgDtqrPAKVym+x6ioB65iiSHW9nWNw8yzyzwIDAQABAoIBAEEw0cPbv6vL5KrF\naMtSY91mwZ3STU6/mQ3VAEOVTi7DtYWFkX+Hx/Vo8JC4btJMfzH/CwQIvzjItIme\nX732yhbrEKoNHGWOXOw1AV97aZqXUl7UTzZvPNQ12Use7k2eoJGQzVxPo0P91519\nu+2MtoW2u54N9tBetrcl8rv0pKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7\nC385ZrjUrpKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7C385ZrjUrpKMhwHoRA+wrrD/7C385Zj\r\n-----END RSA PRIVATE KEY-----\n"
}

pg_conf    = {
    "host": "10.200.51.243",  
    "port": 3306,
    "dbname": "superset",
    "user": "superset_user",
    "password": "FINadmin123#"
}

# ─── FORMULA DICTIONARY & HELPERS ────────────────────────────────────────────
formula_dict = {
    'Net Profit Margin':        'Net Profit / Revenue',
    'Return on Assets (ROA)':    'Net Profit / Total Assets',
    "Return on Equity (ROE)":    "Net Profit / Shareholder's Equity",
    'Return on Capital Employed (ROCE)': 'EBIT / Capital Employed',
    'EBITDA Margin':             'EBITDA / Revenue',
    'Current Ratio':             'Current Assets / Current Liabilities',
    'Quick Ratio (Acid Test)':   '(Current Assets - Inventory) / Current Liabilities',
    'Cash Ratio':                'Cash & Equivalents / Current Liabilities',
    'Inventory Turnover':        'COGS / Average Inventory',
    'Receivables Turnover':      'Revenue / Accounts Receivable',
    'Payables Turnover':         'COGS / Accounts Payable',
    'Asset Turnover':            'Revenue / Total Assets',
    'Working Capital Turnover':  'Revenue / Working Capital',
    'Debt-to-Equity Ratio':      'Total Debt / Shareholder’s Equity',
    'Debt Ratio':                'Total Debt / Total Assets',
    'Interest Coverage Ratio':   'EBIT / Interest Expense',
    'Equity Ratio':              'Equity / Total Assets',
    'Capital Gearing Ratio':     '(Debt / (Debt + Equity))',
    'Earnings Per Share (EPS)':  'Net Income / No. of Shares',
    'Price-to-Earnings (P/E) Ratio': 'Market Price / Earnings Per Share (EPS)',
    'Price-to-Book (P/B) Ratio':     'Market Price / Book Value per Share',
    'Dividend Yield':            'Dividend per Share / Market Price',
    'Dividend Payout Ratio':     'Dividend / Net Profit',
    'Enterprise Value (EV)':     'Market Cap + Debt - Cash',
    'EV/EBITDA':                 'Enterprise Value (EV) / EBITDA',
    'Working Capital':           'Current Assets - Current Liabilities',
    'Gross Profit':              'Revenue - COGS',
    'Free Cash Flow':            'Operating Cash Flow - CapEx',
    'Equity':                    'Assets - Liabilities'
}

OPS = {
    ast.Add:  op.add,
    ast.Sub:  op.sub,
    ast.Mult: op.mul,
    ast.Div:  op.truediv,
    ast.USub: op.neg,
}

def extract_vars_regex(formula_str):
    tokens = re.split(r"[+\-*/()]", formula_str)
    return {tok.strip().rstrip('.') for tok in tokens if tok.strip()}

def resolve_terms(term, seen=None):
    if seen is None:
        seen = set()
    if term in seen:
        raise RuntimeError(f"Cyclic dependency on '{term}'")
    seen.add(term)
    if term not in formula_dict:
        return {term}
    atoms = set()
    for v in extract_vars_regex(formula_dict[term]):
        atoms |= resolve_terms(" ".join(v.split()), seen.copy())
    return atoms

def eval_node(node, vars_):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Name):
        if node.id in vars_:
            return vars_[node.id]
        raise KeyError(f"Unknown var '{node.id}'")
    if isinstance(node, ast.BinOp):
        L = eval_node(node.left, vars_)
        R = eval_node(node.right, vars_)
        return OPS[type(node.op)](L, R)
    if isinstance(node, ast.UnaryOp):
        return OPS[type(node.op)](eval_node(node.operand, vars_))
    raise TypeError(f"Unsupported AST node {node}")

def compute_formula(formula_str, variables):
    tree = ast.parse(formula_str, mode="eval")
    return eval_node(tree.body, variables)

# ─── DATABASE FETCH HELPER ───────────────────────────────────────────────────
def fetch_metric(gid, period_id, nature, scenario):
    sql = f"""
SELECT value FROM "{SCHEMA}"."{TABLE}"
WHERE entity_id={DEFAULT_ENTITY_ID}
  AND grouping_id={gid}
  AND period_id='{period_id}'
  AND nature_of_report='{nature}'
  AND scenario='{scenario}'
  AND taxonomy_id={DEFAULT_TAXONOMY}
  AND reporting_currency='{DEFAULT_CURRENCY}';
"""
    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tf.write(ssh_conf['ssh_pkey']); tf.flush()
    with SSHTunnelForwarder(
        (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
        ssh_username=ssh_conf['ssh_username'],
        ssh_pkey=tf.name,
        remote_bind_address=(pg_conf['host'], pg_conf['port'])
    ) as tunnel:
        conn = (
            f"postgresql://{pg_conf['user']}:{pg_conf['password']}@"
            f"127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
        )
        engine = sqlalchemy.create_engine(conn)
        df     = pd.read_sql(sql, engine)
    if df.empty:
        raise ValueError(f"No data for grouping_id={gid}")
    return float(df['value'].iat[0])

def extract_glossary(nl: str, resources) -> str:
    bi_encoder = resources['bi_encoder']
    term_embs = resources['term_embs']
    term_texts = resources['term_texts']
    reranker_1 = resources['reranker_1']
    q_emb    = bi_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims     = util.cos_sim(q_emb, term_embs)[0]
    idx      = torch.topk(sims, k=TOP_K).indices.tolist()
    tops     = [term_texts[i] for i in idx]
    top_sims = [sims[i].item() for i in idx]
    rerank   = reranker_1.predict([(nl, t) for t in tops])
    scores   = [W_SIM1 * s + W_RERANK1 * r for s, r in zip(top_sims, rerank)]
    best     = int(torch.tensor(scores).argmax().item())
    return tops[best].split(" can be defined as ")[0]

def lookup_grouping(gloss: str, resources):
    bi_encoder = resources['bi_encoder']
    label_embs = resources['label_embs']
    label_texts = resources['label_texts']
    reranker_2 = resources['reranker_2']
    group_df = resources['group_df']
    q_emb    = bi_encoder.encode(gloss, convert_to_tensor=True, normalize_embeddings=True)
    sims     = util.cos_sim(q_emb, label_embs)[0]
    idx      = torch.topk(sims, k=TOP_K).indices.tolist()
    labs     = [label_texts[i] for i in idx]
    lab_sims = [sims[i].item() for i in idx]
    rerank   = reranker_2.predict([(gloss, l) for l in labs])
    scores   = [W_SIM2 * s + W_RERANK2 * r for s, r in zip(lab_sims, rerank)]
    best     = int(torch.tensor(scores).argmax().item())
    lbl      = labs[best]
    gid      = int(group_df.loc[group_df['grouping_label']==lbl,'grouping_id'].iat[0])
    return lbl, gid


# ─── PERIOD RESOLUTION FUNCTIONS ─────────────────────────────────────────
# Build regex for any period unit
months = [m.lower() for m in calendar.month_name if m]
month_regex   = r"(?:{})".format("|".join(months))
quarter_regex = r"(?:q[1-4]|quarter\s*[1-4])"
half_regex    = r"(?:h1|h2|first half|second half|half-year\s*[12])"
fy_regex      = r"(?:fy\s*\d{2,4}|financial year)"
period_unit_regex = rf"{month_regex}|{quarter_regex}|{half_regex}|{fy_regex}"

def detect_view(nl: str, resources) -> str:
    period_encoder = resources['period_encoder']
    view_embs = resources['view_embs']
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

def construct_period_id(nl: str, resources) -> str:
    year   = extract_year(nl)
    nature = extract_nature(nl)
    view   = detect_view(nl, resources)
    seq    = extract_sequence(nl, nature)
    return f"{year}_{nature}_{view}_{seq}"

def model_fn(model_dir, *args):
    """
    1st function called by SageMaker to load the model and any other artifacts.
    Chages to be made:
    - This fn loads only once upon creation of the endpoint. So, 
    we need to change the location of group_df
    - Load Stage 0
    """
    # Reading grouping ID table
    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tf.write(ssh_conf['ssh_pkey']); tf.flush()
    with SSHTunnelForwarder(
        (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
        ssh_username=ssh_conf['ssh_username'],
        ssh_pkey=tf.name,
        remote_bind_address=(pg_conf['host'], pg_conf['port'])
    ) as tunnel:
        conn_str = (
            f"postgresql://{pg_conf['user']}:{pg_conf['password']}@"
            f"127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
        )
        engine   = sqlalchemy.create_engine(conn_str)
        group_df = pd.read_sql(f'SELECT grouping_id, grouping_label FROM "{SCHEMA}".fbi_grouping_master', con=engine)

    # Creating Label Term Infra
    gloss_df = pd.read_csv(os.path.join(model_dir, "data", "glossary.csv"))
    def build_full_text(r):
        txt = f"{r['Glossary']} can be defined as {r['Description']}"
        if pd.notnull(r.get('Formulas, if any')):
            txt += f". Formula: {r['Formulas, if any']}"
        return txt
    term_texts  = gloss_df.apply(build_full_text, axis=1).tolist()
    label_texts = group_df['grouping_label'].tolist()

    # Loading other trained models
    bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
    s1_dir     = os.path.join(model_dir, "models", "stage1_cross_encoder_finetuned_bge_balanced_data_top10")
    s2_dir     = os.path.join(model_dir, "models", "stage2_cross_encoder_finetuned_MiniLM_new_top5")
    device     = 'cuda' if torch.cuda.is_available() else 'cpu' 
    reranker_1 = CrossEncoder(s1_dir, num_labels=1, device=device)
    reranker_2 = CrossEncoder(s2_dir, num_labels=1, device=device)

    # Loading Stage 0
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "models", "stage0_model"), use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # choose device: GPU if available, else CPU
    device = 0 if torch.cuda.is_available() else -1
    # instantiate HF pipeline for text-classification
    stage0_clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    with torch.no_grad():
        term_embs  = bi_encoder.encode(term_texts,  convert_to_tensor=True, normalize_embeddings=True)
        label_embs = bi_encoder.encode(label_texts, convert_to_tensor=True, normalize_embeddings=True)
        view_embs  = bi_encoder.encode(
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

    return {
        "gloss_df":       gloss_df,
        "group_df":       group_df,
        "bi_encoder":     bi_encoder,
        "reranker_1":     reranker_1,
        "reranker_2":     reranker_2,
        "period_encoder": bi_encoder,
        "term_texts":     term_texts,
        "term_embs":      term_embs,
        "label_texts":    label_texts,
        "label_embs":     label_embs,
        "view_embs":      view_embs,
        "stage0_clf": stage0_clf
    }


def input_fn(request_body, content_type="application/json"):
    """
    Deserialize the incoming payload to a Python dictionary.
    """
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, resources):
    """
    Call the handler in inference_log module with the deserialized input.
    input_data should be a dict: {"query": ..., "context": {...}}
    """

    # Load relevant resources
    stage0_clf = resources["stage0_clf"]
    group_df = resources["group_df"]

    # Parse Inputs
    query    = input_data["query"]
    ctx      = input_data["context"]
    scenario = ctx.get("scenario", "Actual")
    if scenario == "Actual":
        scenario = 'Forecast' if 'forecast' in query.lower() or 'budget' in query.lower() else \
                    ('Cashflow' if 'cash' in query.lower() else 'Actual')
    nature   = 'Standalone'

    # 1) Classification + logging
    cls_out   = stage0_clf(query, top_k=None)[0]
    label_idx = int(cls_out["label"].split("_")[-1])
    logger.info(json.dumps({
        "event":           "classification",
        "query":           query,
        "predicted_label": label_idx,
        "score":           cls_out["score"]
    }))

    # 2) Short‑circuit non‑label‑0
    if label_idx != 0:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": (
                    "This question belongs to the type of query we are currently working on; "
                    "we will come back with a solution as soon as possible. "
                    "Thank you for your patience."
                ),
                "predicted_label": label_idx
            })
        }

    # 3) Original inference logic for label 0
    logger.info(json.dumps({
        "event":   "request_received: label 0 complete",
        "query":   query,
        "context": ctx
    }))

    # Stage 1
    gloss = extract_glossary(query, resources)
    logger.info(json.dumps({
        "event":         "stage1_complete",
        "glossary_term": gloss
    }))

    # Logic to check if ratio or not
    label2id = dict(zip(group_df['grouping_label'].str.strip(), group_df['grouping_id']))
    if gloss in formula_dict:
        # Stage 1.5
        atoms     = resolve_terms(gloss)
        logger.info(json.dumps({
            "event":        "formula_resolution",
            "gloss":        gloss,
            "atoms":        list(atoms),
            "grouping_ids": [label2id[t] for t in atoms]
        }))

        period_id = construct_period_id(query, resources)
        logger.info(json.dumps({
            "event":     "period_constructed",
            "period_id": period_id
        }))

        #nature = extract_nature(query)
        sql    = None
        vals   = {t: fetch_metric(label2id[t], period_id, nature, scenario) for t in atoms}
        result = compute_formula(formula_dict[gloss], vals)

    else:
        # Stage 2
        label, gid = lookup_grouping(gloss)
        logger.info(json.dumps({
            "event":          "stage2_complete",
            "glossary_term":  gloss,
            "grouping_label": label,
            "grouping_id":    gid
        }))

        period_id = construct_period_id(query)
        logger.info(json.dumps({
            "event":     "period_constructed",
            "period_id": period_id
        }))

        #nature = extract_nature(query)
        sql    = f"""
SELECT value FROM "{SCHEMA}"."{TABLE}"
WHERE entity_id={DEFAULT_ENTITY_ID}
  AND grouping_id={gid}
  AND period_id='{period_id}'
  AND nature_of_report='{nature}'
  AND scenario='{scenario}'
  AND taxonomy_id={DEFAULT_TAXONOMY}
  AND reporting_currency='{DEFAULT_CURRENCY}';
""".strip()
        logger.info(json.dumps({
            "event": "sql_generated",
            "sql":   sql
        }))

        result = fetch_metric(gid, period_id, nature, scenario)

    logger.info(json.dumps({
        "event": "result_ready",
        "value": result
    }))

    response = {
        "glossary_term": gloss,
        "sql":           sql,
        "period_id":     period_id,
        "value":         result
    }
    if gloss not in formula_dict:
        response.update({
            "grouping_label": label,
            "grouping_id":    gid,
            "nature":         nature,
            "scenario":       scenario
        })

    return response

def output_fn(prediction, accept="application/json"):
    """
    Serialize the prediction output to a JSON string.
    """
    return json.dumps(prediction), accept