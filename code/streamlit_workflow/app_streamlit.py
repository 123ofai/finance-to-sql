import streamlit as st
import pandas as pd
import torch
import tempfile
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from sshtunnel import SSHTunnelForwarder
import sqlalchemy

st.set_page_config(page_title='Finance-to-SQL')

# ─── CONFIG ────────────────────────────────────────────────────────────
# Adjust paths relative to the repo root
GLOSSARY_CSV        = "data/1b_glossary_descriptions.csv"
GROUPING_MASTER_CSV = "data/fbi_grouping_master.csv"

# Schema + table
SCHEMA      = "epm1-replica.finalyzer.info_100032"
TABLE       = "fbi_entity_analysis_report"

# Default report parameters
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"

# Hyperparameters
TOP_K           = 5
W_SIM1, W_RERANK1 = 0.5, 0.5  # Stage1: NL→Glossary
W_SIM2, W_RERANK2 = 0.6, 0.4  # Stage2: Glossary→Grouping

# ─── LOAD SECRETS ────────────────────────────────────────────────────────
ssh_conf = st.secrets["ssh"]
pg_conf  = st.secrets["postgres"]

# ─── CACHE MODELS & DATA ─────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Load data
    gloss_df = pd.read_csv(GLOSSARY_CSV)
    group_df = pd.read_csv(GROUPING_MASTER_CSV)

    # Build enriched glossary texts
    def build_full_text(row):
        txt = f"{row['Glossary']} can be defined as {row['Description']}"
        if pd.notnull(row.get('Formulas, if any')):
            txt += f" Its Formula is: {row['Formulas, if any']}"
        return txt
    term_texts = gloss_df.apply(build_full_text, axis=1).tolist()

    # Initialize models
    bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
    reranker_1 = CrossEncoder('models/stage1_cross_encoder_finetuned_MiniLM_noisyhardnegative_v3_withdesc')
    reranker_2 = CrossEncoder('models/stage2_cross_encoder_finetuned_MiniLM_hardnegative_v2')

    # Precompute embeddings
    with torch.no_grad():
        term_embs  = bi_encoder.encode(term_texts, convert_to_tensor=True, normalize_embeddings=True)
        label_texts = group_df['grouping_label'].tolist()
        label_embs  = bi_encoder.encode(label_texts, convert_to_tensor=True, normalize_embeddings=True)

    return gloss_df, group_df, bi_encoder, reranker_1, reranker_2, term_texts, term_embs, label_texts, label_embs

(gloss_df, group_df,
 bi_encoder, reranker_1, reranker_2,
 term_texts, term_embs,
 label_texts, label_embs) = load_resources()

# ─── PIPELINE FUNCTIONS ─────────────────────────────────────────────────
def extract_glossary(nl_query: str) -> str:
    q_emb = bi_encoder.encode(nl_query, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, term_embs)[0]
    top_idx = torch.topk(sims, k=TOP_K).indices.tolist()
    top_terms = [term_texts[i] for i in top_idx]
    top_sims  = [sims[i].item()  for i in top_idx]

    pairs         = [(nl_query, t) for t in top_terms]
    rerank_scores = reranker_1.predict(pairs)
    final_scores  = [W_SIM1*s + W_RERANK1*r for s, r in zip(top_sims, rerank_scores)]
    best = int(torch.tensor(final_scores).argmax().item())
    return top_terms[best].split(' can be defined as ')[0]


def lookup_grouping(gloss_term: str) -> (str,int):
    q_emb = bi_encoder.encode(gloss_term, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, label_embs)[0]
    top_idx    = torch.topk(sims, k=TOP_K).indices.tolist()
    top_labels = [label_texts[i] for i in top_idx]
    top_sims   = [sims[i].item()  for i in top_idx]

    pairs         = [(gloss_term, lbl) for lbl in top_labels]
    rerank_scores = reranker_2.predict(pairs)
    final_scores  = [W_SIM2*s + W_RERANK2*r for s, r in zip(top_sims, rerank_scores)]
    best = int(torch.tensor(final_scores).argmax().item())
    lbl  = top_labels[best]
    gid  = int(group_df.loc[group_df['grouping_label']==lbl, 'grouping_id'].iat[0])
    return lbl, gid

# ─── AUXILIARY INFERENCE ─────────────────────────────────────────────────
def resolve_period(nl_query: str) -> str:
    return "2024_M_PRD_3"  # stub – replace with dateparser

def infer_scenario(nl_query: str) -> str:
    q = nl_query.lower()
    if 'forecast' in q or 'budget' in q:
        return 'Forecast'
    if 'cash' in q:
        return 'Cashflow'
    return 'Actual'

def infer_nature(nl_query: str) -> str:
    return 'Standalone'

# ─── STREAMLIT LAYOUT ─────────────────────────────────────────────────

st.title('📊 Finance-to-SQL Dashboard')
mode = st.sidebar.selectbox('Mode', ['Run Query', 'Inspect Metrics'])

if mode == 'Inspect Metrics':
    choice = st.selectbox('Metrics to view', ['NL→Glossary', 'Glossary→Grouping'])
    path   = 'results/25May_experiments/stage1_nl2glossary_easydata_rerankerfinetuned.csv' if choice=='NL→Glossary' else 'results/25May_1b/stage2_glossary_to_label_finetunedreranker_weighedscoring.csv'
    df     = pd.read_csv(path)
    st.subheader(f'Viewing {choice} Metrics')
    st.dataframe(df)
else:
    nl_query = st.text_input('Enter your finance question')
    if st.button('Generate & Run'):
        gloss   = extract_glossary(nl_query)
        label, gid = lookup_grouping(gloss)
        period  = resolve_period(nl_query)
        scenario= infer_scenario(nl_query)
        nature  = infer_nature(nl_query)

        params = {
            'entity_id': DEFAULT_ENTITY_ID,
            'grouping_id': gid,
            'period_id': period,
            'nature': nature,
            'scenario': scenario,
            'taxonomy': DEFAULT_TAXONOMY,
            'currency': DEFAULT_CURRENCY
        }
        sql = f"""
SELECT value FROM "{SCHEMA}"."{TABLE}"
WHERE entity_id={params['entity_id']} AND grouping_id={params['grouping_id']} AND period_id='{params['period_id']}'
  AND nature_of_report='{params['nature']}' AND scenario='{params['scenario']}'
  AND taxonomy_id={params['taxonomy']} AND reporting_currency='{params['currency']}';
"""
        st.subheader('🔍 Mapping')
        st.write('**Glossary:**', gloss)
        st.write('**Grouping Label:**', label)
        st.write('**Grouping ID:**', gid)
        st.write('**Period ID:**', period)
        st.write('**Nature:**', nature)
        st.write('**Scenario:**', scenario)

        st.subheader('🛠 Generated SQL')
        st.code(sql, language='sql')

        st.subheader('📈 Results')
        try:
            # write ssh key to temp file
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
                conn = f"postgresql://{pg_conf['user']}:{pg_conf['password']}@127.0.0.1:{lp}/{pg_conf['dbname']}"
                engine = sqlalchemy.create_engine(conn)
                df = pd.read_sql(sql, engine)
                st.dataframe(df)
        except Exception as e:
            st.error(f"SSH/DB error: {e}")
