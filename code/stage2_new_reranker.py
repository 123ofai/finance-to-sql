import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

# 1. Load & split the Stage 2 training data (query=glossary, label=grouping_label, score=0/1)
df = pd.read_csv("../data/stage2_new_reranker_training_data_top10.csv")
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['score']
)

# 2. Wrap as InputExample
train_samples = [
    InputExample(texts=[r['query'], r['label']], label=float(r['score']))
    for _, r in train_df.iterrows()
]
val_samples = [
    InputExample(texts=[r['query'], r['label']], label=float(r['score']))
    for _, r in val_df.iterrows()
]

# 3. Create the evaluator on the validation set
evaluator = CEBinaryClassificationEvaluator.from_input_examples(
    val_samples,
    name="stage2-val"
)

# 4. Initialize your CrossEncoder reranker
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # or your local path
    num_labels=1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 5. Hyperparameters
batch_size   = 16    # stage2 candidate pairs may be shorter, but tune down if OOM
warmup_steps = 50
max_epochs   = 4
min_epochs   = 1    # always run at least this many epochs
patience     = 1    # stop after 1 epoch with no F1 improvement
eps          = 1e-4
save_dir     = "../models/stage2_cross_encoder_finetuned_MiniLM_new_top10"

# 6. Early‐stop trackers
best_val_f1 = float('-inf')
no_improve  = 0

# 7. Epoch-by-epoch training loop
for epoch in range(1, max_epochs + 1):
    print(f"\n=== Stage 2 Reranker Epoch {epoch}/{max_epochs} ===")
    
    # Train exactly one epoch
    model.fit(
        train_dataloader=DataLoader(train_samples, shuffle=True, batch_size=batch_size),
        epochs=1,
        warmup_steps=warmup_steps,
        evaluator=None,
        show_progress_bar=True,
        use_amp=True   # mixed precision to save memory
    )
    
    # Run validation and get metrics dict
    val_results = evaluator(model)
    val_f1      = val_results['stage2-val_f1']
    print(f"Validation F1: {val_f1:.4f}")
    
    # Always accept checkpoints for the first min_epochs
    if epoch <= min_epochs:
        best_val_f1 = val_f1
        no_improve  = 0
        model.save(save_dir)
        print(f" ↳ (Within min_epochs) Saved checkpoint to {save_dir}")
        continue
    
    # Check for meaningful improvement
    if val_f1 > best_val_f1 + eps:
        best_val_f1 = val_f1
        no_improve  = 0
        model.save(save_dir)
        print(f" ↳ Improved F1 → checkpoint saved to {save_dir}")
    else:
        no_improve += 1
        print(f" ↳ No improvement ({no_improve}/{patience})")
    
    # Early stopping
    if no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

print(f"\nBest validation F1: {best_val_f1:.4f}")
print(f"Best model saved in: {save_dir}")
