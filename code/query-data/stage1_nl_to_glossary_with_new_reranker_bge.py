import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─── 1. Load data & models ─────────────────────────────────────────────────────
glossary_df   = pd.read_csv("../data/glossary.csv")
glossary_full = (glossary_df['Glossary'] + ' can be defined as ' + glossary_df['Description']).dropna().tolist()

queries_df    = pd.read_csv("../data/queries.csv")  # your queries CSV

# Bi-encoder
embedder      = SentenceTransformer('BAAI/bge-large-en-v1.5')
term_embeddings = embedder.encode(
    glossary_full,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# Cross-encoder reranker
reranker      = CrossEncoder("../models/stage1_cross_encoder_finetuned_bge_balanced_data_top10")

# ─── 2. Iterate & rerank ────────────────────────────────────────────────────────
results = []
y_true  = []
y_pred  = []

for _, row in queries_df.iterrows():
    query    = row['NL_Query']
    true_lbl = row['GT_Glossary']
    y_true.append(true_lbl)

    # 2.1 Bi-encode + cosine similarity
    q_emb   = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sims    = util.cos_sim(q_emb, term_embeddings)[0]       # shape (N,)
    
    # 2.2 Shortlist top-K by cosine
    K       = 10
    topk_ix = torch.topk(sims, k=K).indices.tolist()
    topk_s  = [sims[i].item() for i in topk_ix]
    topk_txt= [glossary_full[i] for i in topk_ix]

    # 2.3 Rerank those K with cross-encoder
    pairs       = [(query, txt) for txt in topk_txt]
    rerank_scrs = reranker.predict(pairs)                  # list of K floats

    # 2.4 Combine scores (here 50/50)
    final_scrs  = [0.5*s + 0.5*r for s,r in zip(topk_s, rerank_scrs)]
    best_j      = int(torch.tensor(final_scrs).argmax().item())
    
    # 2.5 Extract prediction
    best_full   = topk_txt[best_j]
    pred_term   = best_full.split(' can be defined as ')[0]
    score       = round(final_scrs[best_j], 4)

    # 2.6 Collect
    y_pred.append(pred_term)
    results.append({
        'NL_Query'           : query,
        'GT_Glossary'        : true_lbl,
        'Predicted_Glossary' : pred_term,
        'Combined_Score'     : score
    })

# ─── 3. Build DataFrame, save & inspect ─────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.head(50)  # show top 50 rows in notebook

results_df.to_csv(
    '../results/12june_experiments/stage1_nl2glossary_balanceddata_reranker_bge_finetuned_top5.csv',
    index=False
)

# ─── 4. Compute & print metrics ────────────────────────────────────────────────
acc    = accuracy_score(y_true, y_pred)
f1_mac = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy (Top-1): {acc:.4f}")
print(f"Macro-F1    : {f1_mac:.4f}")
print("\nFull classification report:")
print(classification_report(y_true, y_pred, digits=4))
