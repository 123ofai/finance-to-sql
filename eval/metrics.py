import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load your CSV containing:
#    - 'NL_Query'
#    - 'Original_Glossary' (ground truth)
#    - 'Predicted_Glossary' (from your semantic model)
df = pd.read_csv("/home/gaurav/nl2sql-demo/glossary2label_results.csv")  # Update with your path

# 2. Extract true and predicted labels
y_true = df['ground_label']
y_pred = df['Predicted_Label']

# 3. Compute metrics
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
report = classification_report(y_true, y_pred, digits=4)

# 4. Output results
print(f"Accuracy:   {accuracy:.4f}")
print(f"Macro F₁:   {f1_macro:.4f}")
print(f"Micro F₁:   {f1_micro:.4f}")
print("\nClassification Report:\n", report)
