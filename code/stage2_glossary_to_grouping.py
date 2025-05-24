import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- 1. Glossary to Grouping Label ---
df = pd.read_csv("../data/2_glossary_to_label_gt.csv")

# --- 2. Load grouping_master to get all possible grouping_labels ---
grouping_df = pd.read_csv("../data/fbi_grouping_master.csv")
all_labels = grouping_df["grouping_label"].tolist()

# --- 3. Initialize embedding model ---
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- 4. Embed all grouping_labels once ---
label_embs = model.encode(all_labels, convert_to_tensor=True)

# --- 5. For each predicted glossary term, find best grouping_label ---
predicted_labels = []

for term in df["glossary"]:
    q_emb = model.encode(term, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, label_embs)[0]
    best_idx = torch.argmax(sims).item()
    predicted_labels.append(all_labels[best_idx])

df["predicted_label"] = predicted_labels

df["predicted_grouping_id"] = [
    grouping_df.loc[grouping_df["grouping_label"] == label, "grouping_id"].iat[0]
    if label in grouping_df["grouping_label"].values else None
    for label in df["predicted_label"]
]

# --- 6. Evaluate at label level ---
y_true = df["ground_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_micro = f1_score(y_true, y_pred, average="micro")

print(f"Accuracy (labels): {accuracy:.4f}")
print(f"Macro F1 (labels): {f1_macro:.4f}")
print(f"Micro F1 (labels): {f1_micro:.4f}")
print("\nClassification Report (labels):\n")
print(classification_report(y_true, y_pred, digits=4))

# --- 7. Save enriched CSV ---
path = "../results/24May_refactor/stage2_glossary_to_label.csv"
df.to_csv(path, index=False)
print("✅ Saved " + path)
