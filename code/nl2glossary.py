import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# 1. (Once) Install dependencies in your local environment:
#    pip install pandas sentence-transformers

# 2. Load the glossary
glossary_df = pd.read_csv("/home/gaurav/finalyzer/new_solution/results/glossary_v1.csv")            # your glossary CSV
glossary_terms = glossary_df['Glossary'].dropna().tolist()

# 3. Load your NL queries alongside their ground-truth glossary terms
#    Input CSV must have columns: 'NL_Query' and 'Original_Glossary'
queries_df = pd.read_csv("/home/gaurav/finalyzer/new_solution/results/queries_and_glossary.csv")

# 4. Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 5. Pre-compute embeddings for all glossary terms
term_embeddings = model.encode(glossary_terms, convert_to_tensor=True)

# 6. Iterate through each query and find the best matching glossary term
results = []
for _, row in queries_df.iterrows():
    query_text = row['query']
    original = row['glossary_term']
    
    # Embed the query
    q_emb = model.encode(query_text, convert_to_tensor=True)
    
    # Compute cosine similarity to all glossary terms
    sims = util.cos_sim(q_emb, term_embeddings)
    
    # Identify best match
    best_idx = torch.argmax(sims).item()
    predicted = glossary_terms[best_idx]
    score = sims[0][best_idx].item()
    
    results.append({
        'NL_Query': query_text,
        'GT_Glossary': original,
        'Predicted_Glossary': predicted,
        'Similarity_Score': round(score, 4)
    })

# 7. Convert to DataFrame and save or inspect
results_df = pd.DataFrame(results)
print(results_df.head(10))
results_df.to_csv("semantic_comparison_results.csv", index=False)
# Save the results to a CSV file
results_df.to_csv('/home/gaurav/Downloads/nl2glossary.csv', index=False)  # Update with your path
