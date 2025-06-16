import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import CrossEncoder
import numpy as np


# 2. Load the glossary
glossary_df = pd.read_csv("../data/glossary.csv")            # your glossary CSV

# Method 1
glossary_terms = glossary_df['Glossary']
glossary_terms = glossary_terms.dropna().tolist()

# Method 2
glossary_full = glossary_df['Glossary'] + ' can be defined as '+ glossary_df['Description']
glossary_full = glossary_full.dropna().tolist()

# 3. Load the queries
queries_df = pd.read_csv("../data/queries.csv")
queries_df = queries_df.dropna(subset=['GT_Glossary'])  # your queries CSV

# 4. Load the model
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 5. Pre-compute embeddings for all glossary terms
term_embeddings = model.encode(glossary_full, convert_to_tensor=True, normalize_embeddings=True)

rows = []
for _, row in queries_df.iterrows():
    query   = row['NL_Query']
    gt      = row['GT_Glossary']

    # find the index of the term itself
    true_idx = glossary_terms.index(gt)

    # embed & shortlist
    q_emb    = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sims     = util.cos_sim(q_emb, term_embeddings)[0]
    top5_idx = torch.topk(sims, k=10).indices.tolist()

    # ensure GT is included
    if true_idx not in top5_idx:
        top5_idx.pop(-1)
        top5_idx.insert(0, true_idx)

    # 1 positive + 4 negatives
    for idx in top5_idx:
        candidate = glossary_full[idx]
        score     = 1.0 if idx == true_idx else 0.0
        rows.append({
            'query': query,
            'label': candidate,
            'score': score
        })

# save CSV
train_df = pd.DataFrame(rows)
train_df.to_csv("../data/reranker_train_top10.csv", index=False)
