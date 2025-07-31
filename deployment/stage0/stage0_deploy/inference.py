# inference.py

import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def model_fn(model_dir, *args):
    """
    Called once when the container starts.
    Load tokenizer & model from model_dir (the SageMaker model artifact location).
    """
    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # choose device: GPU if available, else CPU
    device = 0 if torch.cuda.is_available() else -1
    # instantiate HF pipeline for text-classification
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return clf

def input_fn(serialized_input, content_type):
    """
    Deserialize and extract the query string from the request.
    Expect JSON: {"query": "<your text here>"}
    """
    if content_type == "application/json":
        data = json.loads(serialized_input)
        return data["query"]
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, clf):
    """
    Run the pipeline on the deserialized input and return raw predictions.
    """
    # returns list of dicts: [{"label":"LABEL_0","score":0.95},...]
    return clf(input_data, top_k=None)

def output_fn(prediction, accept):
    """
    Serialize the prediction result back to JSON.
    """
    return json.dumps(prediction), "application/json"
