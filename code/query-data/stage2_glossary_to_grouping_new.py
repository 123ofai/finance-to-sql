import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from sklearn.metrics import accuracy_score, f1_score

# ─── 1. Load & merge glossary ↔ descriptions ↔ ground truth ────────────────
df = pd.read_csv("../data/2_glossary_to_label_gt_new.csv")  # columns: Glossary, ground_label
desc_df = pd.read_csv("../data/1b_glossary_descriptions.csv")
df_merged = df.merge(
    desc_df[['Glossary', 'Description']],
    on='Glossary', how='left'
)

# ─── 2. Load and clean grouping_master ───────────────────────────────────────
grouping_df = pd.read_csv("../data/fbi_grouping_master.csv")
# Normalize whitespace in grouping labels
grouping_df['grouping_label_clean'] = (
    grouping_df['grouping_label']
      .astype(str)
      .str.strip()
      .str.replace(r'\s+', ' ', regex=True)
)

# Build a dict: cleaned label → grouping_id
label2id = dict(zip(
    grouping_df['grouping_label_clean'],
    grouping_df['grouping_id']
))

# Master list of cleaned labels for embedding
all_labels = grouping_df['grouping_label_clean'].tolist()

# ─── 3. Initialize models ───────────────────────────────────────────────────
# Bi-encoder for fast embedding
embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Cross-encoder reranker
reranker = CrossEncoder(
    "../models/stage2_cross_encoder_finetuned_MiniLM_new_top10",
    num_labels=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# ─── 4. Precompute all label embeddings ─────────────────────────────────────
label_embs = embedder.encode(
    all_labels,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# ─── 5. Inference loop: glossary → grouping_label ───────────────────────────
predicted_labels = []
for _, row in df_merged.iterrows():
    # Normalize glossary term as query
    query = row['Glossary'].strip()

    # 5.1 Embed & cosine-sim shortlist
    q_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, label_embs)[0]

    top_k = 10
    topk_idx = torch.topk(sims, k=top_k).indices.tolist()
    topk_labels = [all_labels[i] for i in topk_idx]
    topk_sims = [sims[i].item() for i in topk_idx]

    # 5.2 Rerank with CrossEncoder
    pairs = [(query, lbl) for lbl in topk_labels]
    rerank_scores = reranker.predict(pairs)

    # 5.3 Fuse scores and pick best
    final_scores = [0.6 * sim + 0.4 * rer for sim, rer in zip(topk_sims, rerank_scores)]
    best_i = int(torch.tensor(final_scores).argmax().item())
    predicted_labels.append(topk_labels[best_i])

# Attach predicted_label
df_merged['predicted_label'] = predicted_labels

# ─── 6. Map predicted_label → predicted_grouping_id ────────────────────────
def lookup_id(lbl: str):
    if pd.isna(lbl):
        return None
    clean = ' '.join(str(lbl).strip().split())
    return label2id.get(clean)

df_merged['predicted_grouping_id'] = df_merged['predicted_label'].apply(lookup_id)

# ─── 7. Evaluate at label level ─────────────────────────────────────────────
y_true = df_merged['ground_label'].str.strip().str.replace(r'\s+', ' ', regex=True)
y_pred = df_merged['predicted_label']

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy (labels): {accuracy:.4f}")
print(f"Macro F1   (labels): {f1_macro:.4f}")

# ─── 8. Save enriched CSV ───────────────────────────────────────────────────
out_path = "../results/12june_experiments/stage2_glossary_to_label_finetunedreranker_new_top10.csv"
df_merged.to_csv(out_path, index=False)
print("✅ Saved", out_path)
