import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import os

# Force CPU-only environment
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

# 1. Load & split
df = pd.read_csv("../data/reranker_training_data_top10.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['score'])

# 2. Wrap as InputExample
train_samples = [InputExample(texts=[r['query'], r['label']], label=float(r['score'])) 
                 for _, r in train_df.iterrows()]
val_samples   = [InputExample(texts=[r['query'], r['label']], label=float(r['score'])) 
                 for _, r in val_df.iterrows()]

# 3. Evaluator
evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples, name="validation")

# 4. Model
model = CrossEncoder("BAAI/bge-reranker-v2-m3", num_labels=1, device="cuda", max_length= 128)

# 5. Hyperparameters
batch_size   = 16
warmup_steps = 100
max_epochs   = 10
min_epochs   = 2
patience     = 1
eps          = 1e-4

# 6. Early‐stop trackers for F1
best_val_f1 = float('-inf')
no_improve  = 0

# 7. Training loop
for epoch in range(1, max_epochs+1):
    print(f"\n=== Epoch {epoch}/{max_epochs} ===")
    # Train one epoch
    model.fit(
        train_dataloader=DataLoader(train_samples, shuffle=True, batch_size=batch_size),
        epochs=1,
        warmup_steps=warmup_steps,
        evaluator=None,
        show_progress_bar=True,
        use_amp= False
    )

    # Validate
    val_results = evaluator(model)
    val_f1 = val_results['validation_f1']
    print(f"Validation F1: {val_f1:.4f}")

    # Always accept within the minimum epochs
    if epoch <= min_epochs:
        best_val_f1 = val_f1
        no_improve  = 0
        model.save("../models/stage1_cross_encoder_finetuned_bgm_balanced_data_top5")
        print(" ↳ (Within min_epochs) saved new best checkpoint.")
        continue

    # Check for meaningful improvement
    if val_f1 > best_val_f1 + eps:
        best_val_f1 = val_f1
        no_improve  = 0
        model.save("../models/stage1_cross_encoder_finetuned_bgm_balanced_data_top10")
        print(" ↳ Improved F1! checkpoint saved.")
    else:
        no_improve += 1
        print(f" ↳ No improvement ({no_improve}/{patience}).")

    # Early stopping
    if no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch}.")
        break

print(f"\nBest validation F1: {best_val_f1:.4f}")
print("Best model is saved in ../models/stage1_cross_encoder_finetuned_bgm_balaced_data_top10")
