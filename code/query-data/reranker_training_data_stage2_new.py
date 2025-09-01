import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# 1. Load your ground‐truth and master label list
gt_df        = pd.read_csv("../data/2_glossary_to_label_gt_new.csv")
master_df    = pd.read_csv("../data/fbi_grouping_master.csv")

# Remove leading/trailing spaces (and collapse multiple spaces to one)
master_df["grouping_label"] = (
    master_df["grouping_label"]
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

all_labels   = master_df["grouping_label"].tolist()

gt_df["ground_label"] = (
    gt_df["ground_label"]
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

# 2. (Optional) load descriptions if you want to include them in the “query”
# desc_df      = pd.read_csv("../data/1b_glossary_descriptions.csv")
# gt_df        = gt_df.merge(
#     desc_df[["Glossary","Description"]],
#     on="Glossary", how="left"
# )

# 3. Bi‐encoder for embedding labels
bi_encoder   = SentenceTransformer("BAAI/bge-large-en-v1.5")
label_embs   = bi_encoder.encode(
    all_labels,
    convert_to_tensor=True,
    normalize_embeddings=True
)

rows = []
K    = 10

for _, row in gt_df.iterrows():
    term       = row["Glossary"]
    desc       = row.get("Description", "")
    query_text = term  # or f"{term} can be defined as {desc}"

    # 3.1 Embed the glossary term
    q_emb = bi_encoder.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
    sims  = util.cos_sim(q_emb, label_embs)[0]  # shape (L,)

    # 3.2 Get top-K
    topk_idx = torch.topk(sims, k=K).indices.tolist()

    # 3.3 Ensure the true label is in there
    true_lbl = row["ground_label"]
    true_idx = all_labels.index(true_lbl)
    if true_idx not in topk_idx:
        topk_idx.pop(-1)
        topk_idx.insert(0, true_idx)

    # 3.4 Record 1 positive + K–1 negatives
    for idx in topk_idx:
        candidate = all_labels[idx]
        score     = 1.0 if idx == true_idx else 0.0
        rows.append({
            "query": query_text,
            "label": candidate,
            "score": score
        })

# 4. Save to CSV
train2_df = pd.DataFrame(rows)
train2_df.to_csv("../data/stage2_new_reranker_training_data_top10.csv", index=False)
print(f"Saved {len(train2_df)} examples to 3b_stage2_reranker_training_data_hard_negatives.csv")
