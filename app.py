import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sshtunnel import SSHTunnelForwarder
import sqlalchemy

# ─── CONFIG ────────────────────────────────────────────────────────────
GLOSSARY_CSV = "glossary_v1.csv"
GROUPING_CSV = "fbi_grouping_master.csv"
METRIC_CSV1  = "nl2glossary.csv"              # NL→Glossary metrics CSV
METRIC_CSV2  = "glossary2label_results.csv"   # Glossary→Grouping metrics CSV
EMBED_MODEL  = "sentence-transformers/all-mpnet-base-v2"

# Schema + table
SCHEMA      = "epm1-replica.finalyzer.info_100032"
TABLE       = "fbi_entity_analysis_report"

# Default report parameters
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"

# ─── LOAD SECRETS ────────────────────────────────────────────────────────
# In Streamlit Cloud “Settings → Secrets”, define two top-level tables: [ssh] and [postgres]
ssh = st.secrets["ssh"]
pg  = st.secrets["postgres"]

# ─── CACHE DATA & MODELS ─────────────────────────────────────────────────
@st.cache_resource
def load_models_and_data():
    gloss_df    = pd.read_csv(GLOSSARY_CSV)
    group_df    = pd.read_csv(GROUPING_CSV)
    model       = SentenceTransformer(EMBED_MODEL)

    gloss_terms = gloss_df["Glossary"].tolist()
    gloss_embs  = model.encode(gloss_terms, convert_to_tensor=True)

    group_lbls  = group_df["grouping_label"].tolist()
    group_embs  = model.encode(group_lbls, convert_to_tensor=True)

    return gloss_df, group_df, model, gloss_terms, gloss_embs, group_lbls, group_embs

gloss_df, group_df, model, gloss_terms, gloss_embs, group_lbls, group_embs = load_models_and_data()

# ─── WORKFLOW FUNCTIONS ─────────────────────────────────────────────────
def extract_glossary(nl: str) -> str:
    emb  = model.encode(nl, convert_to_tensor=True)
    sims = util.cos_sim(emb, gloss_embs)[0]
    return gloss_terms[int(torch.argmax(sims))]

def lookup_grouping(term: str):
    emb  = model.encode(term, convert_to_tensor=True)
    sims = util.cos_sim(emb, group_embs)[0]
    idx  = int(torch.argmax(sims))
    return group_lbls[idx], int(group_df.loc[idx, "grouping_id"])

def resolve_period(nl: str) -> str:
    # TODO: implement real period resolution
    return "2024_M_PRD_3"

def infer_scenario(nl: str) -> str:
    return "Cashflow" if "cash" in nl.lower() else "Actual"

def infer_nature(nl: str) -> str:
    return "Standalone"

# ─── Define your SQL template WITH NO f-prefix ───────────────────────────────
SQL_TMPL = """
SELECT value
FROM "{SCHEMA}"."{TABLE}"
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
    # First, substitute SCHEMA and TABLE (this could also be hard-coded if you prefer)
    sql = SQL_TMPL.format(SCHEMA=SCHEMA, TABLE=TABLE, **params)
    return sql


# ─── STREAMLIT UI ───────────────────────────────────────────────────────
st.title("📊 Finance-to-SQL Dashboard")

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

        # Build SQL
        params = {
            "entity_id":   DEFAULT_ENTITY_ID,
            "grouping_id": gid,
            "period_id":   period,
            "nature":      nature,
            "scenario":    scenario,
            "taxonomy":    DEFAULT_TAXONOMY,
            "currency":    DEFAULT_CURRENCY
        }
        sql = make_sql(params)

        # Display mapping & SQL
        st.subheader("🔍 Mapping Steps")
        st.write("**Natural Query:**", nl_query)
        st.write("**Glossary Term:**", gloss)
        st.write("**Grouping Label:**", label)
        st.write("**Grouping ID:**", gid)
        st.write("**Period ID:**", period)
        st.write("**Nature:**", nature)
        st.write("**Scenario:**", scenario)

        st.subheader("🛠 Generated SQL")
        st.code(sql, language="sql")

        # 4) Execute via SSH tunnel (Streamlit Cloud)
        st.subheader("📈 Query Results")
        try:
            ssh_conf = st.secrets["ssh"]
            pg_conf  = st.secrets["postgres"]

            with SSHTunnelForwarder(
                (ssh_conf["tunnel_host"], ssh_conf["tunnel_port"]),
                ssh_username = ssh_conf["ssh_username"],
                ssh_pkey     = ssh_conf["ssh_pkey"].encode(),
                remote_bind_address = (pg_conf["host"], pg_conf["port"])
        ) as tunnel:
            local_port = tunnel.local_bind_port
            conn_str   = (
                f"postgresql://{pg_conf['user']}:{pg_conf['password']}"
                f"@127.0.0.1:{local_port}/{pg_conf['dbname']}"
            )
            engine  = sqlalchemy.create_engine(conn_str)
            results = pd.read_sql(sql, engine)
            st.dataframe(results)

        except Exception as e:
            st.error(f"SSH/DB error: {e}")

