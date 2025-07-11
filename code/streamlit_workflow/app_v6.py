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

# ─── 0. Stage 1.5: Formula dictionary & helpers ────────────────────────────
formula_dict = {
    "Net Profit Margin":       "Net Profit / Revenue",
    "Capital Gearing Ratio":   "(Debt / (Debt + Equity))",
    "Enterprise Value (EV)":   "Market Cap + Debt - Cash",
    "Quick Ratio (Acid Test)": "(Current Assets - Inventory) / Current Liabilities",
    # … add all your formula‐bearing terms …
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

def eval_node(n, vars_):
    if isinstance(n, ast.Num):
        return n.n
    if isinstance(n, ast.Name):
        if n.id in vars_:
            return vars_[n.id]
        raise KeyError(f"Unknown var '{n.id}'")
    if isinstance(n, ast.BinOp):
        L, R = eval_node(n.left, vars_), eval_node(n.right, vars_)
        return OPS[type(n.op)](L, R)
    if isinstance(n, ast.UnaryOp):
        return OPS[type(n.op)](eval_node(n.operand, vars_))
    raise TypeError(f"Unsupported AST node {n}")

def compute_formula(formula_str, variables):
    tree = ast.parse(formula_str, mode="eval")
    return eval_node(tree.body, variables)


# ─── CONFIG & constants ─────────────────────────────────────────────────────
GLOSSARY_CSV        = "data/1b_glossary_descriptions.csv"
SCHEMA              = "epm1-replica.finalyzer.info_100032"
TABLE               = "fbi_entity_analysis_report"
DEFAULT_ENTITY_ID   = 6450
DEFAULT_TAXONOMY    = 71
DEFAULT_CURRENCY    = "INR"

TOP_K             = 5
W_SIM1, W_RERANK1 = 0.5, 0.5
W_SIM2, W_RERANK2 = 0.6, 0.4


# ─── LOAD SECRETS ───────────────────────────────────────────────────────────
ssh_conf   = st.secrets["ssh"]
pg_conf    = st.secrets["postgres"]
azure_conf = st.secrets["azure"]


# ─── AZURE‐BLOB DOWNLOAD ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_model_folder_from_blob(prefix: str) -> str:
    client    = BlobServiceClient.from_connection_string(azure_conf["connection_string"])
    container = client.get_container_client(azure_conf["container_name"])
    tmpdir    = tempfile.mkdtemp(prefix="azblob_")
    for blob in container.list_blobs(name_starts_with=prefix):
        if blob.name.endswith("/"):
            continue
        rel       = os.path.relpath(blob.name, prefix)
        local_path= os.path.join(tmpdir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data      = container.get_blob_client(blob).download_blob().readall()
        with open(local_path, "wb") as f:
            f.write(data)
    return tmpdir


# ─── CACHE MODELS & DATA ────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # 1) Glossary from CSV
    gloss_df = pd.read_csv(GLOSSARY_CSV)

    # 2) grouping_master from Postgres over SSH
    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tf.write(ssh_conf['ssh_pkey']); tf.flush()
    with SSHTunnelForwarder(
        (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
        ssh_username=ssh_conf['ssh_username'],
        ssh_pkey=tf.name,
        remote_bind_address=(pg_conf['host'], pg_conf['port'])
    ) as tunnel:
        conn_str = (
            f"postgresql://{pg_conf['user']}:{pg_conf['password']}"
            f"@127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
        )
        engine  = sqlalchemy.create_engine(conn_str)
        group_df = pd.read_sql(
            'SELECT grouping_id, grouping_label FROM fbi_grouping_master',
            con=engine
        )

    # 3) Build enriched glossary texts
    def build_full_text(r):
        txt = f"{r['Glossary']} can be defined as {r['Description']}"
        if pd.notnull(r.get('Formulas, if any')):
            txt += f". Formula: {r['Formulas, if any']}"
        return txt

    term_texts  = gloss_df.apply(build_full_text, axis=1).tolist()
    label_texts = group_df['grouping_label'].tolist()

    # 4) Load models
    bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

    base_pref = azure_conf["model_prefix"]
    s1_pref   = base_pref + "stage1_cross_encoder_finetuned_bge_balanced_data_top10"
    s2_pref   = base_pref + "stage2_cross_encoder_finetuned_MiniLM_new_top5"
    s1_dir    = download_model_folder_from_blob(s1_pref)
    s2_dir    = download_model_folder_from_blob(s2_pref)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    reranker_1= CrossEncoder(s1_dir, num_labels=1, device=device)
    reranker_2= CrossEncoder(s2_dir, num_labels=1, device=device)

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

# Build label→id dict for fast lookup
label2id = dict(zip(
    group_df['grouping_label'].str.strip(),
    group_df['grouping_id']
))


# ─── STAGE 1 & STAGE 2 FUNCTIONS ─────────────────────────────────────────────
def extract_glossary(nl: str) -> str:
    q_emb     = bi_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims      = util.cos_sim(q_emb, term_embs)[0]
    idx       = torch.topk(sims, k=TOP_K).indices.tolist()
    tops      = [term_texts[i] for i in idx]
    top_sims  = [sims[i].item() for i in idx]
    rerank    = reranker_1.predict([(nl, t) for t in tops])
    scores    = [W_SIM1*s + W_RERANK1*r for s, r in zip(top_sims, rerank)]
    best      = int(torch.tensor(scores).argmax().item())
    return tops[best].split(' can be defined as ')[0]


def lookup_grouping(gloss: str) -> (str, int):
    q_emb     = bi_encoder.encode(gloss, convert_to_tensor=True, normalize_embeddings=True)
    sims      = util.cos_sim(q_emb, label_embs)[0]
    idx       = torch.topk(sims, k=TOP_K).indices.tolist()
    labs      = [label_texts[i] for i in idx]
    lab_sims  = [sims[i].item() for i in idx]
    rerank    = reranker_2.predict([(gloss, l) for l in labs])
    scores    = [W_SIM2*s + W_RERANK2*r for s, r in zip(lab_sims, rerank)]
    best      = int(torch.tensor(scores).argmax().item())
    lbl       = labs[best]
    gid       = int(group_df.loc[group_df['grouping_label'] == lbl, 'grouping_id'].iat[0])
    return lbl, gid


# ─── PERIOD HANDLING & FETCH METRIC ──────────────────────────────────────────
# (reuse your existing detect_view, extract_year, extract_nature, extract_sequence, construct_period_id)
# plus the fetch_metric(...) function unchanged


# ─── STREAMLIT LAYOUT ───────────────────────────────────────────────────────
st.title('📊 Finance-to-SQL Dashboard')
mode = st.sidebar.selectbox('Mode', ['Run Query', 'Inspect Metrics'])

if mode == 'Inspect Metrics':
    choice = st.selectbox('Metrics to view', ['NL→Glossary', 'Glossary→Grouping'])
    path   = (
        'results/stage1_nl2glossary_easydata_rerankerfinetuned.csv'
        if choice == 'NL→Glossary'
        else 'results/stage2_glossary_to_label_finetunedreranker_weighedscoring.csv'
    )
    st.dataframe(pd.read_csv(path), height=600)

else:
    nl_query = st.text_input('Enter your finance question:')
    if st.button('Generate & Run'):
        # Stage 1
        gloss = extract_glossary(nl_query)

        # Stage 1.5
        if gloss in formula_dict:
            atoms     = resolve_terms(gloss)
            period_id = construct_period_id(nl_query)
            nature    = extract_nature(nl_query)
            scenario  = ('Forecast' if 'forecast' in nl_query.lower() else
                         'Cashflow' if 'cash' in nl_query.lower() else 'Actual')
            vals = {
                t: fetch_metric(label2id[t], period_id, nature, scenario)
                for t in atoms
            }
            result = compute_formula(formula_dict[gloss], vals)
            st.subheader(f"📐 Computed **{gloss}**")
            st.write(f"Formula: `{formula_dict[gloss]}`")
            st.write(f"Value: **{result}**")
            st.stop()

        # Stage 2
        label, gid = lookup_grouping(gloss)
        period_id  = construct_period_id(nl_query)
        nature     = extract_nature(nl_query)
        scenario   = ('Forecast' if 'forecast' in nl_query.lower() else
                      'Cashflow' if 'cash' in nl_query.lower() else 'Actual')

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
                conn   = (
                    f"postgresql://{pg_conf['user']}:{pg_conf['password']}"
                    f"@127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
                )
                engine = sqlalchemy.create_engine(conn)
                df_resp= pd.read_sql(sql, engine)
                st.dataframe(df_resp)
        except Exception as e:
            st.error(f"SSH/DB error: {e}")
