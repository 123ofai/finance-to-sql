import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score

# ─── 1. Load & merge glossary ↔ descriptions ↔ ground truth ────────────────
df = pd.read_csv("../data/2_glossary_to_label_gt_new.csv")  # Glossary, ground_label
desc = pd.read_csv("../data/1b_glossary_descriptions.csv")  # Glossary, Description
df = df.merge(desc[['Glossary','Description']], on='Glossary', how='left')

# Normalize your ground_truth labels
df['ground_label'] = (
    df['ground_label']
      .astype(str)
      .str.strip()
      .str.replace(r'\s+', ' ', regex=True)
)

# ─── 2. Load & clean grouping_master ─────────────────────────────────────────
grouping_df = pd.read_csv("../data/fbi_grouping_master.csv")
grouping_df['grouping_label'] = (
    grouping_df['grouping_label']
      .astype(str)
      .str.strip()
      .str.replace(r'\s+', ' ', regex=True)
)
all_labels = grouping_df['grouping_label'].tolist()

# (Optional) build a dict for mapping label→ID later
label2id = dict(zip(grouping_df['grouping_label'], grouping_df['grouping_id']))

# ─── 3. Initialize bi-encoder & precompute embeddings ───────────────────────
embedder   = SentenceTransformer('BAAI/bge-large-en-v1.5')
label_embs = embedder.encode(
    all_labels,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# ─── 4. Retrieval: top-1 & top-5 ────────────────────────────────────────────
records = []
for _, row in df.iterrows():
    query = row['Glossary'].strip()
    gt    = row['ground_label']

    # Embed & cosine-sim
    q_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, label_embs)[0]

    # Top-5 indices
    top5_idx    = torch.topk(sims, k=10).indices.tolist()
    top5_labels = [all_labels[i] for i in top5_idx]

    # Top-1 is the first of those
    top1_label = top5_labels[0]

    records.append({
        'Glossary'    : query,
        'Ground_Label': gt,
        'Top1'        : top1_label,
        'Top5'        : top5_labels
    })

res_df = pd.DataFrame(records)

# ─── 5. Compute metrics ──────────────────────────────────────────────────────
y_true = res_df['Ground_Label']
y_top1 = res_df['Top1']

acc_top1   = accuracy_score(y_true, y_top1)
f1_top1    = f1_score(y_true, y_top1, average='macro')
recall_at5 = res_df.apply(lambda r: int(r['Ground_Label'] in r['Top5']), axis=1).mean()

print(f"Top-1 Accuracy   : {acc_top1:.4f}")
print(f"Top-1 Macro-F1   : {f1_top1:.4f}")
print(f"Recall@5         : {recall_at5:.4f}")

# ─── 6. (Optional) Map Top1 to grouping_id ───────────────────────────────────
def lookup_id(lbl):
    return label2id.get(lbl.strip())

res_df['Top1_Grouping_ID'] = res_df['Top1'].apply(lookup_id)
res_df['GT_Grouping_ID']   = res_df['Ground_Label'].apply(lookup_id)

# ─── 7. Save results ─────────────────────────────────────────────────────────
out_path = "../results/stage2_bi_encoder_top1_top5_cleaned.csv"
#res_df.to_csv(out_path, index=False)
print("✅ Saved results to", out_path)
