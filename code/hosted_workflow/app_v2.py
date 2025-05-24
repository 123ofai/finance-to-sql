#!/usr/bin/env python3
# finance_to_sql_local_schema.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sshtunnel import SSHTunnelForwarder
import sqlalchemy

# ─── CONFIG ────────────────────────────────────────────────────────────
# Data files
GLOSSARY_CSV = "glossary_v1.csv"
GROUPING_CSV = "fbi_grouping_master.csv"
PEM_FILE     = "bsq-devops-3rdparty-vendor.pem"

# Embedding model
EMBED_MODEL  = "sentence-transformers/all-mpnet-base-v2"

# SSH / Jump-host (edit to your values)
SSH_TUNNEL_HOST = "3.110.31.32"  # replace
SSH_TUNNEL_PORT = 22
SSH_USER        = "ec2-user"

# Private Postgres (edit to your values)
DB_HOST     = "10.200.51.243"       # replace
DB_PORT     = 3306
DB_NAME     = "superset"
DB_USER     = "superset_user"
DB_PASSWORD = "FINadmin123#"  # replace

# Schema-qualified table
SCHEMA      = "epm1-replica.finalyzer.info_100032"
TABLE       = "fbi_entity_analysis_report"

# Default report parameters
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"

# ─── LOAD & PREPARE ─────────────────────────────────────────────────────
glossary_df = pd.read_csv(GLOSSARY_CSV)
grouping_df = pd.read_csv(GROUPING_CSV)

model        = SentenceTransformer(EMBED_MODEL)
gloss_terms  = glossary_df["Glossary"].tolist()
gloss_embs   = model.encode(gloss_terms, convert_to_tensor=True)
group_labels = grouping_df["grouping_label"].tolist()
group_embs   = model.encode(group_labels, convert_to_tensor=True)

# ─── WORKFLOW FUNCTIONS ─────────────────────────────────────────────────
def extract_glossary(nl: str) -> str:
    emb  = model.encode(nl, convert_to_tensor=True)
    sims = util.cos_sim(emb, gloss_embs)[0]
    return gloss_terms[int(torch.argmax(sims))]

def lookup_grouping(term: str):
    emb  = model.encode(term, convert_to_tensor=True)
    sims = util.cos_sim(emb, group_embs)[0]
    idx  = int(torch.argmax(sims))
    return group_labels[idx], int(grouping_df.loc[idx, "grouping_id"])

def resolve_period(nl: str) -> str:
    # Replace with real logic; stub uses last month
    return "2024_M_PRD_3"

def infer_scenario(nl: str) -> str:
    return "Cashflow" if "cash" in nl.lower() else "Actual"

def infer_nature(nl: str) -> str:
    return "Standalone"

SQL_TMPL = f"""
SELECT value
FROM "{SCHEMA}"."{TABLE}"
WHERE entity_id          = {{entity_id}}
  AND grouping_id        = {{grouping_id}}
  AND period_id          = '{{period_id}}'
  AND nature_of_report   = '{{nature}}'
  AND scenario           = '{{scenario}}'
  AND taxonomy_id        = {{taxonomy}}
  AND reporting_currency = '{{currency}}'
;
"""

def make_sql(params: dict) -> str:
    return SQL_TMPL.format(**params)

# ─── MAIN INTERACTIVE LOOP ────────────────────────────────────────────────
def main():
    print("\n📊 Finance-to-SQL Demo (Local w/ Schema & SSH)\n(Type 'exit' to quit)\n")
    while True:
        nl_query = input("Enter your finance question> ").strip()
        if nl_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # 1) NL → Glossary
        gloss = extract_glossary(nl_query)
        # 2) Glossary → Grouping
        label, gid = lookup_grouping(gloss)
        # 3) Period, Scenario, Nature
        period   = resolve_period(nl_query)
        scenario = infer_scenario(nl_query)
        nature   = infer_nature(nl_query)

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
        print("\n🔍 Mapping Steps")
        print(f"  NL Query       : {nl_query}")
        print(f"  Glossary Term  : {gloss}")
        print(f"  Grouping Label : {label}")
        print(f"  Grouping ID    : {gid}")
        print(f"  Period ID      : {period}")
        print(f"  Scenario       : {scenario}")
        print(f"  Nature         : {nature}")
        print("\n🛠 Generated SQL:")
        print(sql)

        # 4) SSH Tunnel + Execute
        print("\n🔐 Opening SSH tunnel...")
        try:
            with SSHTunnelForwarder(
                (SSH_TUNNEL_HOST, SSH_TUNNEL_PORT),
                ssh_username=SSH_USER,
                ssh_pkey=PEM_FILE,
                remote_bind_address=(DB_HOST, DB_PORT)
            ) as tunnel:
                lp = tunnel.local_bind_port
                print(f"✅ Tunnel open: localhost:{lp} → {DB_HOST}:{DB_PORT}")

                conn_str = (
                    f"postgresql://{DB_USER}:{DB_PASSWORD}"
                    f"@127.0.0.1:{lp}/{DB_NAME}"
                )
                engine = sqlalchemy.create_engine(conn_str)
                print("🔗 Running query...")
                df = pd.read_sql(sql, engine)
                print("\n📈 Query Results:")
                print(df.to_string(index=False))
        except Exception as e:
            print(f"❌ SSH/DB error: {e}")

        print("\n" + "-"*70 + "\n")

if __name__ == "__main__":
    main()
