import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sshtunnel import SSHTunnelForwarder
import sqlalchemy

# ─── CONFIG ────────────────────────────────────────────────────────────
GLOSSARY_CSV = "glossary_v1.csv"
GROUPING_CSV = "fbi_grouping_master.csv"
METRIC_CSV1  = "nl2glossary.csv"             # NL→Glossary CSV
METRIC_CSV2  = "glossary2label_results.csv"  # Glossary→Grouping CSV
EMBED_MODEL  = "sentence-transformers/all-mpnet-base-v2"

# Load secrets
# In Streamlit Cloud secrets, define [ssh] and [postgres] sections
ssh = st.secrets["ssh"]
pg  = st.secrets["postgres"]

# ─── CACHE DATA & MODELS ──────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    gloss_df    = pd.read_csv(GLOSSARY_CSV)
    group_df    = pd.read_csv(GROUPING_CSV)
    model       = SentenceTransformer(EMBED_MODEL)
    gloss_terms = gloss_df["Glossary"].tolist()
    gloss_embs  = model.encode(gloss_terms, convert_to_tensor=True)
    group_lbls  = group_df["grouping_label"].tolist()
    group_embs  = model.encode(group_lbls, convert_to_tensor=True)
    return gloss_df, group_df, model, gloss_terms, gloss_embs, group_lbls, group_embs

gloss_df, group_df, model, gloss_terms, gloss_embs, group_lbls, group_embs = load_resources()

# ─── WORKFLOW FUNCTIONS ─────────────────────────────────────────────────
def extract_glossary(nl: str) -> str:
    emb  = model.encode(nl, convert_to_tensor=True)
    sims = util.cos_sim(emb, gloss_embs)[0]
    return gloss_terms[int(torch.argmax(sims))]

def lookup_grouping(term: str):
    emb  = model.encode(term, convert_to_tensor=True)
    sims = util.cos_sim(emb, group_embs)[0]
    idx  = int(torch.argmax(sims))
    return group_lbls[idx], group_df.loc[idx, "grouping_id"]

def resolve_period(nl: str) -> str:
    # TODO: Replace with real period-resolver logic
    return "2024_M_PRD_3"

def infer_scenario(nl: str) -> str:
    return "Cashflow" if "cash" in nl.lower() else "Actual"

def infer_nature(nl: str) -> str:
    return "Standalone"

SQL_TMPL = """
SELECT value
FROM fbi_entity_analysis_report
WHERE entity_id          = {entity_id}
  AND grouping_id        = {grouping_id}
  AND period_id          = '{period_id}'
  AND nature_of_report   = '{nature}'
  AND scenario           = '{scenario}'
  AND taxonomy_id        = {taxonomy}
  AND reporting_currency = '{currency}'
;
"""
def make_sql(params: dict) -> str:
    return SQL_TMPL.format(**params)

# ─── STREAMLIT UI ───────────────────────────────────────────────────────
st.title("📊 Finance-to-SQL & Metrics Dashboard")

mode = st.sidebar.selectbox("Mode", ["Run Query", "Inspect Metrics"])

if mode == "Inspect Metrics":
    choice = st.selectbox("Which metrics to view?", ["NL→Glossary", "Glossary→Grouping"])
    csv    = METRIC_CSV1 if choice == "NL→Glossary" else METRIC_CSV2
    df     = pd.read_csv(csv)
    st.subheader(f"Viewing: {choice} Metrics")
    st.dataframe(df, height=600)

else:  # Run Query
    nl_query = st.text_input("Enter your finance question:")
    if st.button("Generate & Run"):
        # 1) NL → Glossary
        gloss = extract_glossary(nl_query)
        # 2) Glossary → Grouping
        label, gid = lookup_grouping(gloss)
        # 3) Period, Scenario, Nature
        period   = resolve_period(nl_query)
        scenario = infer_scenario(nl_query)
        nature   = infer_nature(nl_query)

        # 4) Build SQL
        params = {
            "entity_id":    6450,
            "grouping_id":  gid,
            "period_id":    period,
            "nature":       nature,
            "scenario":     scenario,
            "taxonomy":     71,
            "currency":     "INR"
        }
        sql = make_sql(params)

        # 5) Display mapping
        st.subheader("🔍 Mapping Steps")
        st.write("**Glossary Term:**", gloss)
        st.write("**Grouping Label:**", label)
        st.write("**Grouping ID:**", gid)
        st.write("**Period ID:**", period)
        st.write("**Scenario:**", scenario)
        st.write("**Nature:**", nature)

        # 6) Show generated SQL
        st.subheader("🛠 Generated SQL")
        st.code(sql, language="sql")

        # 7) Execute via SSH Tunnel
        st.subheader("📈 Query Results")
        try:
            with SSHTunnelForwarder(
                (ssh["tunnel_host"], ssh["tunnel_port"]),
                ssh_username = ssh["ssh_username"],
                ssh_pkey     = ssh["ssh_pkey"].encode(),
                remote_bind_address=(pg["host"], pg["port"])
            ) as tunnel:
                local_port = tunnel.local_bind_port
                conn_str   = (
                    f"postgresql://{pg['user']}:{pg['password']}"
                    f"@127.0.0.1:{local_port}/{pg['dbname']}"
                )
                engine = sqlalchemy.create_engine(conn_str)
                results = pd.read_sql(sql, engine)
                st.dataframe(results)
        except Exception as e:
            st.error(f"Error running query: {e}")
