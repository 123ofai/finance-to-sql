# app.py

import os
import re
import calendar
import tempfile
import ast
import operator as op
from datetime import datetime

import streamlit as st
import pandas as pd
import torch
import sqlalchemy
from sshtunnel import SSHTunnelForwarder
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rapidfuzz import process, fuzz

st.set_page_config(page_title='Finance-to-SQL')

# ─── CONFIG ────────────────────────────────────────────────────────────
GLOSSARY_CSV        = "data/1b_glossary_descriptions.csv"
SCHEMA              = "epm1-replica.finalyzer.info_100032"
TABLE               = "fbi_entity_analysis_report"
DEFAULT_ENTITY_ID   = 6450
DEFAULT_TAXONOMY    = 71
DEFAULT_CURRENCY    = "INR"
TOP_K               = 5
W_SIM1, W_RERANK1   = 0.5, 0.5
W_SIM2, W_RERANK2   = 0.6, 0.4

# ─── 0. Stage 1.5: formula dict & helpers ──────────────────────────────
formula_dict = {
    'Net Profit Margin': 'Net Profit / Revenue', 
    'Return on Assets (ROA)': 'Net Profit / Total Assets', 
    'Return on Equity (ROE)': "Net Profit / Shareholder's Equity", 
    'Return on Capital Employed (ROCE)': 'EBIT / Capital Employed', 
    'EBITDA Margin': 'EBITDA / Revenue', 
    'Current Ratio': 'Current Assets / Current Liabilities', 
    'Quick Ratio (Acid Test)': '(Current Assets - Inventory) / Current Liabilities', 
    'Cash Ratio': 'Cash & Equivalents / Current Liabilities', 
    'Inventory Turnover': 'COGS / Average Inventory', 
    'Receivables Turnover': 'Revenue / Accounts Receivable', 
    'Payables Turnover': 'COGS / Accounts Payable', 
    'Asset Turnover': 'Revenue / Total Assets', 
    'Working Capital Turnover': 'Revenue / Working Capital', 
    'Debt-to-Equity Ratio': 'Total Debt / Shareholder’s Equity', 
    'Debt Ratio': 'Total Debt / Total Assets', 
    'Interest Coverage Ratio': 'EBIT / Interest Expense', 
    'Equity Ratio': 'Equity / Total Assets', 
    'Capital Gearing Ratio': '(Debt / (Debt + Equity))', 
    'Earnings Per Share (EPS)': 'Net Income / No. of Shares', 
    'Price-to-Earnings (P/E) Ratio': 'Market Price / Earnings Per Share (EPS)', 
    'Price-to-Book (P/B) Ratio': 'Market Price / Book Value per Share', 
    'Dividend Yield': 'Dividend per Share / Market Price', 
    'Dividend Payout Ratio': 'Dividend / Net Profit', 
    'Enterprise Value (EV)': 'Market Cap + Debt - Cash', 
    'EV/EBITDA': 'Enterprise Value (EV) / EBITDA', 
    'Working Capital': 'Current Assets - Current Liabilities.', 
    'Gross Profit': 'Revenue - COGS.', 
    'Free Cash Flow': 'Operating Cash Flow - CapEx', 
    'Equity': 'Assets - Liabilities'
}

def extract_vars_regex(formula_str):
    tokens = re.split(r"[+\-*/\(\)]", formula_str)
    return {tok.strip().rstrip('.') for tok in tokens if tok.strip()}

def resolve_terms(term, seen=None):
    if seen is None: seen = set()
    if term in seen:
        raise RuntimeError(f"Cyclic dependency on '{term}'")
    seen.add(term)
    if term not in formula_dict:
        return {term}
    atoms = set()
    for v in extract_vars_regex(formula_dict[term]):
        atoms |= resolve_terms(" ".join(v.split()), seen.copy())
    return atoms

OPS = {
    ast.Add:  op.add,
    ast.Sub:  op.sub,
    ast.Mult: op.mul,
    ast.Div:  op.truediv,
    ast.USub: op.neg,
}

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

# ─── LOAD SECRETS ───────────────────────────────────────────────────────
ssh_conf   = st.secrets["ssh"]
pg_conf    = st.secrets["postgres"]
azure_conf = st.secrets["azure"]

# ─── MODEL DOWNLOAD ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_model_folder_from_blob(prefix: str) -> str:
    client    = BlobServiceClient.from_connection_string(azure_conf["connection_string"])
    container = client.get_container_client(azure_conf["container_name"])
    tmpdir    = tempfile.mkdtemp(prefix="azblob_")
    for blob in container.list_blobs(name_starts_with=prefix):
        if blob.name.endswith("/"): continue
        rel       = os.path.relpath(blob.name, prefix)
        local_path= os.path.join(tmpdir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data      = container.get_blob_client(blob).download_blob().readall()
        with open(local_path, "wb") as f: f.write(data)
    return tmpdir

# ─── CACHE & LOAD RESOURCES ──────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # 1) Glossary from CSV
    gloss_df = pd.read_csv(GLOSSARY_CSV)

    # 2) grouping_master via SSH
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

    # 3) Build enriched texts
    def build_full_text(r):
        txt = f"{r['Glossary']} can be defined as {r['Description']}"
        if pd.notnull(r.get('Formulas, if any')):
            txt += f". Formula: {r['Formulas, if any']}"
        return txt

    term_texts  = gloss_df.apply(build_full_text, axis=1).tolist()
    label_texts = group_df['grouping_label'].tolist()

    # 4) Load models
    bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
    base_pref  = azure_conf["model_prefix"]
    s1_dir     = download_model_folder_from_blob(base_pref + "stage1_cross_encoder_finetuned_bge_balanced_data_top10")
    s2_dir     = download_model_folder_from_blob(base_pref + "stage2_cross_encoder_finetuned_MiniLM_new_top5")
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    reranker_1 = CrossEncoder(s1_dir, num_labels=1, device=device)
    reranker_2 = CrossEncoder(s2_dir, num_labels=1, device=device)

    # 5) Precompute embeddings
    with torch.no_grad():
        term_embs  = bi_encoder.encode(term_texts, convert_to_tensor=True, normalize_embeddings=True)
        label_embs = bi_encoder.encode(label_texts, convert_to_tensor=True, normalize_embeddings=True)

    # 6) Period encoder
    period_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

    return (
        gloss_df, group_df,
        bi_encoder, reranker_1, reranker_2,
        term_texts, term_embs,
        label_texts, label_embs,
        period_encoder
    )

(
    gloss_df, group_df,
    bi_encoder, reranker_1, reranker_2,
    term_texts, term_embs,
    label_texts, label_embs,
    period_encoder
) = load_resources()

# label → id map
label2id = dict(zip(
    group_df['grouping_label'].str.strip(),
    group_df['grouping_id']
))

# ─── STAGE 1 & STAGE 2 FUNCTIONS ─────────────────────────────────────────
def extract_glossary(nl: str) -> str:
    q_emb    = bi_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims     = util.cos_sim(q_emb, term_embs)[0]
    idx      = torch.topk(sims, k=TOP_K).indices.tolist()
    tops     = [term_texts[i] for i in idx]
    top_sims = [sims[i].item() for i in idx]
    rerank   = reranker_1.predict([(nl, t) for t in tops])
    scores   = [W_SIM1*s + W_RERANK1*r for s, r in zip(top_sims, rerank)]
    best     = int(torch.tensor(scores).argmax().item())
    return tops[best].split(' can be defined as ')[0]

def lookup_grouping(gloss: str) -> (str,int):
    q_emb    = bi_encoder.encode(gloss, convert_to_tensor=True, normalize_embeddings=True)
    sims     = util.cos_sim(q_emb, label_embs)[0]
    idx      = torch.topk(sims, k=TOP_K).indices.tolist()
    labs     = [label_texts[i] for i in idx]
    lab_sims = [sims[i].item() for i in idx]
    rerank   = reranker_2.predict([(gloss, l) for l in labs])
    scores   = [W_SIM2*s + W_RERANK2*r for s, r in zip(lab_sims, rerank)]
    best     = int(torch.tensor(scores).argmax().item())
    lbl      = labs[best]
    gid      = int(group_df.loc[group_df['grouping_label']==lbl,'grouping_id'].iat[0])
    return lbl, gid

# ─── PERIOD RESOLUTION FUNCTIONS ─────────────────────────────────────────
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

# Build regex for any period unit
months = [m.lower() for m in calendar.month_name if m]
month_regex   = r"(?:{})".format("|".join(months))
quarter_regex = r"(?:q[1-4]|quarter\s*[1-4])"
half_regex    = r"(?:h1|h2|first half|second half|half-year\s*[12])"
fy_regex      = r"(?:fy\s*\d{2,4}|financial year)"
period_unit_regex = rf"{month_regex}|{quarter_regex}|{half_regex}|{fy_regex}"

# “Verbose” prototypes for semantic fallback
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

# Precompute embeddings for view prototypes
view_embs = period_encoder.encode(
    VIEW_PROTOS, 
    convert_to_tensor=True, 
    normalize_embeddings=True
)

def detect_view(nl: str) -> str:
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

    best_proto = VIEW_PROTOS[best_idx]
    return "FTP" if best_proto in FTP_PROTOS_VERB else "PRD"


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

def extract_sequence(nl: str, nat: str) -> int:
    low = nl.lower()

    # 1) EXPLICIT MONTH
    if nat == "M":
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
    if nat == "FQ":
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
    if nat == "FH":
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
    if nat == "FY":
        return 1

    # 5) FALLBACK: current period
    month = datetime.now().month
    if nat == "M":
        return month
    if nat == "FQ":
        return (month - 1)//3 + 1
    if nat == "FH":
        return 1 if month <= 6 else 2
    return 1

def construct_period_id(nl: str) -> str:
    year   = extract_year(nl)
    period_nature = extract_nature(nl)
    view   = detect_view(nl)
    seq    = extract_sequence(nl, nature)
    return f"{year}_{period_nature}_{view}_{seq}"


def fetch_metric(gid, period_id, scenario):
    sql = f"""
SELECT value FROM "{SCHEMA}"."{TABLE}"
WHERE entity_id={DEFAULT_ENTITY_ID}
  AND grouping_id={gid}
  AND period_id='{period_id}'
  AND nature_of_report='Standalone'
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
        conn   = f"postgresql://{pg_conf['user']}:{pg_conf['password']}@127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
        engine = sqlalchemy.create_engine(conn)
        df     = pd.read_sql(sql, engine)
    if df.empty:
        raise ValueError(f"No data for grouping_id={gid}")
    return float(df['value'].iat[0])

# ─── STREAMLIT LAYOUT ─────────────────────────────────────────────────────
st.title('📊 Finance-to-SQL Dashboard')
mode = st.sidebar.selectbox('Mode', ['Run Query', 'Inspect Metrics'])

if mode == 'Inspect Metrics':
    choice = st.selectbox('Metrics to view', ['NL→Glossary', 'Glossary→Grouping'])
    path   = (
        'results/stage1_nl2glossary.csv' 
        if choice=='NL→Glossary' 
        else 'results/stage2_glossary_to_label.csv'
    )
    st.dataframe(pd.read_csv(path), height=600)

else:
    nl_query = st.text_input('Enter your finance question:')
    if st.button('Generate & Run'):
        # Stage 1
        gloss = extract_glossary(nl_query)

        # Stage 1.5 (formula case)
        if gloss in formula_dict:
            atoms     = resolve_terms(gloss)
            period_id = construct_period_id(nl_query)
            nature    = "Standalone"
            scenario  = (
                'Forecast' if 'forecast' in nl_query.lower()
                else 'Cashflow' if 'cash' in nl_query.lower()
                else 'Actual'
            )

            # build vals by looking up each atom’s grouping_id
            vals = {}
            for atom in atoms:
                # find the best grouping label & its ID for this atom
                grouping_label, gid = lookup_grouping(atom)
                # fetch the metric value
                vals[atom] = fetch_metric(
                    gid,
                    period_id,
                    nature,
                    scenario
                )

            result = compute_formula(formula_dict[gloss], vals)

            st.subheader(f"📐 Computed **{gloss}**")
            st.write(f"Formula: `{formula_dict[gloss]}`")
            st.write(f"Value: **{result}**")
            st.stop()

        # Stage 2 (non-formula lookup)
        label, gid = lookup_grouping(gloss)
        period_id  = construct_period_id(nl_query)
        nature     = "Standalone"
        scenario   = (
            'Forecast' if 'forecast' in nl_query.lower()
            else 'Cashflow' if 'cash' in nl_query.lower()
            else 'Actual'
        )

        sql = f"""
SELECT value FROM "{SCHEMA}"."{TABLE}"
WHERE entity_id={DEFAULT_ENTITY_ID}
  AND grouping_id={gid}
  AND period_id='{period_id}'
  AND nature_of_report='Standalone'
  AND scenario='{scenario}'
  AND taxonomy_id={DEFAULT_TAXONOMY}
  AND reporting_currency='{DEFAULT_CURRENCY}';
"""
        st.subheader('🔍 Mapping')
        st.write('**Glossary Term:**', gloss)
        st.write('**Grouping Label:**', label)
        st.write('**Grouping ID:**', gid)
        st.write('**Period ID:**', period_id)
        st.write('**Nature:**', nature)
        st.write('**Scenario:**', scenario)

        st.subheader('🛠 Generated SQL')
        st.code(sql, language='sql')

        st.subheader('📈 Query Results')
        try:
            tf = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            tf.write(ssh_conf['ssh_pkey']); tf.flush()
            with SSHTunnelForwarder(
                (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
                ssh_username=ssh_conf['ssh_username'],
                ssh_pkey=tf.name,
                remote_bind_address=(pg_conf['host'], pg_conf['port'])
            ) as tunnel:
                conn   = f"postgresql://{pg_conf['user']}:{pg_conf['password']}@127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
                engine = sqlalchemy.create_engine(conn)
                df_rsp= pd.read_sql(sql, engine)
                st.dataframe(df_rsp)
        except Exception as e:
            st.error(f"SSH/DB error: {e}")
