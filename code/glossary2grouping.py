import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- 1. Load your 100-query CSV (NL_Query + Original_Glossary) ---
df = pd.read_csv("/home/gaurav/finalyzer/new_solution/glossary2label_matching.csv")
# Ensure columns: 'NL_Query', 'Original_Glossary', 'Predicted_Glossary'

# --- 2. Load grouping_master to get all possible grouping_labels ---
grouping_df = pd.read_csv("/home/gaurav/finalyzer/new_data/fbi_grouping_master.csv")
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
    for label in df["Predicted_Label"]
]

# --- 6. Evaluate at label level ---
y_true = df["ground_label"]
y_pred = df["Predicted_Label"]

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_micro = f1_score(y_true, y_pred, average="micro")

print(f"Accuracy (labels): {accuracy:.4f}")
print(f"Macro F1 (labels): {f1_macro:.4f}")
print(f"Micro F1 (labels): {f1_micro:.4f}")
print("\nClassification Report (labels):\n")
print(classification_report(y_true, y_pred, digits=4))

# --- 7. Save enriched CSV ---
df.to_csv("/home/gaurav/finalyzer/new_solution/results/glossary2label_results.csv", index=False)
print("✅ Saved to semantic_label_matching_results.csv")
