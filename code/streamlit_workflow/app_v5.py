import os
import re
import calendar
import tempfile
from datetime import datetime

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from sshtunnel import SSHTunnelForwarder
import sqlalchemy
from rapidfuzz import process, fuzz
import streamlit as st
from azure.storage.blob import BlobServiceClient

st.set_page_config(page_title='Finance-to-SQL')

# ─── CONFIG ────────────────────────────────────────────────────────────
GLOSSARY_CSV        = "data/1b_glossary_descriptions.csv"
GROUPING_MASTER_CSV = "data/fbi_grouping_master.csv"
SCHEMA      = "epm1-replica.finalyzer.info_100032"
TABLE       = "fbi_entity_analysis_report"
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"
TOP_K             = 5
W_SIM1, W_RERANK1 = 0.5, 0.5
W_SIM2, W_RERANK2 = 0.6, 0.4

# ─── LOAD SECRETS ────────────────────────────────────────────────────────
ssh_conf   = st.secrets["ssh"]
pg_conf    = st.secrets["postgres"]
azure_conf = st.secrets["azure"]

# ─── AZURE-BLOB DOWNLOAD HELPER ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_model_folder_from_blob(prefix: str) -> str:
    conn_str       = azure_conf["connection_string"]
    container_name = azure_conf["container_name"]
    client         = BlobServiceClient.from_connection_string(conn_str)
    container      = client.get_container_client(container_name)

    tmpdir = tempfile.mkdtemp(prefix="azblob_")
    for blob in container.list_blobs(name_starts_with=prefix):
        if blob.name.endswith("/"):
            continue
        rel_path   = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data = container.get_blob_client(blob).download_blob().readall()
        with open(local_path, "wb") as f:
            f.write(data)

    return tmpdir

# ─── CACHE MODELS & DATA ─────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # 1) DataFrames
    gloss_df = pd.read_csv(GLOSSARY_CSV)
    group_df = pd.read_csv(GROUPING_MASTER_CSV)

    # 2) Enriched texts
    def build_full_text(r):
        txt = f"{r['Glossary']} can be defined as {r['Description']}"
        if pd.notnull(r.get('Formulas, if any')):
            txt += f" Its Formula is: {r['Formulas, if any']}"
        return txt

    term_texts  = gloss_df.apply(build_full_text, axis=1).tolist()
    label_texts = group_df['grouping_label'].tolist()

    # 3) Bi-encoder
    bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

    # 4) Download fine-tuned rerankers from Blob
    base_prefix   = azure_conf["model_prefix"]
    s1_prefix     = base_prefix + "stage1_cross_encoder_finetuned_bge_balanced_data_top10"
    s2_prefix     = base_prefix + "stage2_cross_encoder_finetuned_MiniLM_new_top5"
    stage1_dir    = download_model_folder_from_blob(s1_prefix)
    stage2_dir    = download_model_folder_from_blob(s2_prefix)

    reranker_1 = CrossEncoder.from_pretrained(stage1_dir)
    reranker_2 = CrossEncoder.from_pretrained(stage2_dir)

    # 5) Precompute embeddings
    with torch.no_grad():
        term_embs  = bi_encoder.encode(term_texts, convert_to_tensor=True, normalize_embeddings=True)
        label_embs = bi_encoder.encode(label_texts, convert_to_tensor=True, normalize_embeddings=True)

    # 6) Period‐view encoder
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

# ─── PIPELINE: NL→Glossary & Glossary→Grouping ───────────────────────────
def extract_glossary(nl: str) -> str:
    q_emb = bi_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, term_embs)[0]
    top_idx    = torch.topk(sims, k=TOP_K).indices.tolist()
    top_terms  = [term_texts[i] for i in top_idx]
    top_sims   = [sims[i].item() for i in top_idx]
    rerank     = reranker_1.predict([(nl, t) for t in top_terms])
    scores     = [W_SIM1*s + W_RERANK1*r for s, r in zip(top_sims, rerank)]
    best       = int(torch.tensor(scores).argmax().item())
    return top_terms[best].split(' can be defined as ')[0]

def lookup_grouping(gloss: str) -> (str,int):
    q_emb = bi_encoder.encode(gloss, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, label_embs)[0]
    top_idx    = torch.topk(sims, k=TOP_K).indices.tolist()
    top_labels = [label_texts[i] for i in top_idx]
    top_sims   = [sims[i].item() for i in top_idx]
    rerank     = reranker_2.predict([(gloss, lbl) for lbl in top_labels])
    scores     = [W_SIM2*s + W_RERANK2*r for s, r in zip(top_sims, rerank)]
    best       = int(torch.tensor(scores).argmax().item())
    lbl        = top_labels[best]
    gid        = int(group_df.loc[group_df['grouping_label']==lbl, 'grouping_id'].iat[0])
    return lbl, gid

# ─── PERIOD HANDLING ──────────────────────────────────────────────────────
FTP_KEYWORDS = [r"\bfor the period\b", r"\bfor that period\b", r"\bjust that month\b",
                r"\bonly that quarter\b", r"\bas at\b", r"\bas at\s+(?:month|quarter)\b"]
PRD_KEYWORDS = [r"\byear to date\b", r"\bytd\b", r"\bso far\b", r"\bcumulative\b",
                r"\bthrough\b", r"\bup to\b", r"\bas of\b", r"\bto date\b",
                r"\bsince the start of the year\b", r"\bmonth to date\b", r"\bmtd\b",
                r"\bquarter to date\b", r"\bqtd\b", r"\bthrough end of\b",
                r"\bthrough end-of-period\b"]

months         = [m.lower() for m in calendar.month_name if m]
month_regex    = rf"(?:{'|'.join(months)})"
quarter_regex  = r"(?:q[1-4]|quarter\s*[1-4])"
half_regex     = r"(?:h1|h2|first half|second half|half-year\s*[12])"
fy_regex       = r"(?:fy\s*\d{2,4}|financial year)"
period_unit_rx = rf"{month_regex}|{quarter_regex}|{half_regex}|{fy_regex}"

FTP_PROTOS = [
    "FTP can be defined as ‘for the period’ meaning only that month or quarter",
    "FTP can be defined as ‘for that period only’ meaning the single slice of time",
    "FTP can be defined as ‘at month end’ meaning only that month",
    "FTP can be defined as ‘year ended’ meaning the year‐end snapshot",
]
PRD_PROTOS = [
    "PRD can be defined as ‘to date’ meaning cumulative up until now",
    "PRD can be defined as ‘year to date’ meaning aggregated so far this fiscal year",
    "PRD can be defined as ‘month to date’ meaning cumulative this month",
    "PRD can be defined as ‘quarter to date’ meaning cumulative this quarter",
    "PRD can be defined as ‘so far’ meaning sum of all periods up until date",
]
VIEW_PROTOS = FTP_PROTOS + PRD_PROTOS
view_embs   = period_encoder.encode(VIEW_PROTOS, convert_to_tensor=True, normalize_embeddings=True)

def detect_view(nl: str) -> str:
    low = nl.lower()
    for pat in PRD_KEYWORDS:
        if re.search(pat, low):
            return "PRD"
    if re.search(rf"\bfor\b.*\b{period_unit_rx}\b", low):
        return "FTP"
    q_emb = period_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, view_embs)[0]
    best_idx, best_score = int(sims.argmax()), sims.max().item()
    if best_score < 0.45:
        return "FTP"
    return "FTP" if VIEW_PROTOS[best_idx] in FTP_PROTOS else "PRD"

month_full   = [m.lower() for m in calendar.month_name if m]
month_abbr   = [m[:3].lower() for m in calendar.month_name if m]
all_months   = set(month_full + month_abbr)
quarter_caps = ["q1","q2","q3","q4","quarter 1","quarter 2","quarter 3","quarter 4",
                "1st quarter","2nd quarter","3rd quarter","4th quarter"]
half_caps    = ["h1","h2","first half","second half","half-year 1","half-year 2"]
ORDINAL_MAP  = {"first":1,"1st":1,"second":2,"2nd":2,"third":3,"3rd":3,"fourth":4,"4th":4}
NUM_WORD_MAP = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
                "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}

def fuzzy_match_token(token, candidates, threshold=75):
    match, score, _ = process.extractOne(token, candidates, scorer=fuzz.ratio)
    return match if score >= threshold else None

def extract_year(nl: str) -> int:
    m = re.search(r"\b(19|20)\d{2}\b", nl)
    return int(m.group()) if m else datetime.now().year

def extract_nature(nl: str) -> str:
    low = nl.lower()
    if re.search(r"\bquarter\b|\bq[1-4]\b", low): return "FQ"
    if re.search(r"\bhalf\b|\bh1\b|\bh2\b",   low): return "FH"
    if re.search(r"\bfinancial year\b|\bfy\b", low): return "FY"
    return "M"

def extract_sequence(nl: str, nature: str) -> int:
    low = nl.lower()
    now = datetime.now()
    # MONTH
    if nature == "M":
        for i, m in enumerate(calendar.month_name[1:], start=1):
            if m.lower() in low or m.lower()[:3] in low:
                return i
        for w,v in NUM_WORD_MAP.items():
            if re.search(rf"\b{w}\b", low): return v
        m = re.search(r"month\s+(\d+)", low)
        if m and 1 <= int(m.group(1)) <= 12: return int(m.group(1))
        for t in re.findall(r"\w+", low):
            fm = fuzzy_match_token(t, all_months)
            if fm:
                return month_full.index(fm)+1 if fm in month_full else month_abbr.index(fm)+1
        if "last month" in low:
            return max(1, now.month-1)
        m = re.search(r"last\s+(\d+)\s+months?", low)
        if m: return max(1, now.month - int(m.group(1)))
    # QUARTER
    if nature == "FQ":
        m = re.search(r"q([1-4])", low)
        if m: return int(m.group(1))
        for w,v in ORDINAL_MAP.items():
            if f"{w} quarter" in low: return v
        for t in re.findall(r"\w+", low):
            fm = fuzzy_match_token(t, quarter_caps)
            if fm: return int(re.search(r"([1-4])", fm).group(1))
        if "last quarter" in low:
            cur_q = (now.month-1)//3 + 1
            return max(1, cur_q-1)
        m = re.search(r"last\s+(\d+)\s+quarters?", low)
        if m:
            cur_q = (now.month-1)//3 + 1
            return max(1, cur_q - int(m.group(1)))
    # HALF
    if nature == "FH":
        if re.search(r"h1\b|first half", low): return 1
        if re.search(r"h2\b|second half", low): return 2
        for w,v in ORDINAL_MAP.items():
            if f"{w} half" in low: return v
        for t in re.findall(r"\w+", low):
            fm = fuzzy_match_token(t, half_caps)
            if fm: return 1 if "1" in fm else 2
        return 1 if now.month <= 6 else 2
    # FY
    if nature == "FY":
        return 1
    # fallback
    if nature == "M":  return now.month
    if nature == "FQ": return (now.month-1)//3 + 1
    if nature == "FH": return 1 if now.month <= 6 else 2
    return 1

def construct_period_id(nl: str) -> str:
    y     = extract_year(nl)
    nat   = extract_nature(nl)
    view  = detect_view(nl)
    seq   = extract_sequence(nl, nat)
    return f"{y}_{nat}_{view}_{seq}"

# ─── STREAMLIT LAYOUT ─────────────────────────────────────────────────────
st.title('📊 Finance-to-SQL Dashboard')
mode = st.sidebar.selectbox('Mode', ['Run Query', 'Inspect Metrics'])

if mode == 'Inspect Metrics':
    choice = st.selectbox('Metrics to view', ['NL→Glossary', 'Glossary→Grouping'])
    path   = (
        'results/25May_experiments/stage1_nl2glossary_easydata_rerankerfinetuned.csv'
        if choice == 'NL→Glossary'
        else 'results/25May_1b/stage2_glossary_to_label_finetunedreranker_weighedscoring.csv'
    )
    dfm = pd.read_csv(path)
    st.subheader(f'Viewing: {choice} Metrics')
    st.dataframe(dfm, height=600)

else:
    nl_query = st.text_input('Enter your finance question:')
    if st.button('Generate & Run'):
        gloss     = extract_glossary(nl_query)
        label, gid= lookup_grouping(gloss)
        period_id = construct_period_id(nl_query)
        scenario  = ('Forecast' if any(k in nl_query.lower() for k in ['forecast','budget'])
                     else ('Cashflow' if 'cash' in nl_query.lower() else 'Actual'))
        nature    = 'Standalone'

        params = {
            'entity_id':   DEFAULT_ENTITY_ID,
            'grouping_id': gid,
            'period_id':   period_id,
            'nature':      nature,
            'scenario':    scenario,
            'taxonomy':    DEFAULT_TAXONOMY,
            'currency':    DEFAULT_CURRENCY
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
            tf.write(ssh_conf['ssh_pkey'])
            tf.flush()
            with SSHTunnelForwarder(
                (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
                ssh_username=ssh_conf['ssh_username'],
                ssh_pkey=tf.name,
                remote_bind_address=(pg_conf['host'], pg_conf['port'])
            ) as tunnel:
                lp = tunnel.local_bind_port
                conn = (
                    f"postgresql://{pg_conf['user']}:{pg_conf['password']}"
                    f"@127.0.0.1:{lp}/{pg_conf['dbname']}"
                )
                engine  = sqlalchemy.create_engine(conn)
                df_resp = pd.read_sql(sql, engine)
                st.dataframe(df_resp)
        except Exception as e:
            st.error(f"SSH/DB error: {e}")
