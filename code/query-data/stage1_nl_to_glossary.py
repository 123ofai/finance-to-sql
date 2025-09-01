import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 2. Load the glossary
glossary_df = pd.read_csv("../data/1b_glossary_descriptions.csv")            # your glossary CSV
glossary_terms = glossary_df['Glossary'].dropna().tolist()

# 3. Load your NL queries alongside their ground-truth glossary terms
#    Input CSV must have columns: 'NL_Query' and 'GT_Glossary'
queries_df = pd.read_csv("../data/1a_nl_to_glossary_gt.csv")

# 4. Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 5. Pre-compute embeddings for all glossary terms
term_embeddings = model.encode(glossary_terms, convert_to_tensor=True)

# 6. Iterate through each query and find the best matching glossary term
results = []
y_true = []
y_pred = []
for _, row in queries_df.iterrows():
    query_text = row['NL_Query']
    original = row['GT_Glossary']
    
    # Embed the query
    q_emb = model.encode(query_text, convert_to_tensor=True)
    
    # Compute cosine similarity to all glossary terms
    sims = util.cos_sim(q_emb, term_embeddings)
    
    # Identify best match
    best_idx = torch.argmax(sims).item()
    predicted = glossary_terms[best_idx]
    score = sims[0][best_idx].item()

    # For metric calc
    y_true.append(original)
    y_pred.append(predicted)
    
    results.append({
        'NL_Query': query_text,
        'GT_Glossary': original,
        'Predicted_Glossary': predicted,
        'Similarity_Score': round(score, 4)
    })

# 7. Convert to DataFrame and save or inspect
results_df = pd.DataFrame(results)
print(results_df.head(5))

# 8. Evaluate
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")

print(f"Accuracy (labels): {accuracy:.4f}")
print(f"Macro F1 (labels): {f1_macro:.4f}")
#print("\nClassification Report (labels):\n")
#print(classification_report(y_true, y_pred, digits=4))

# Save the results to a CSV file
results_df.to_csv('../results/24May_refactor/stage1_nl2glossary.csv', index=False)