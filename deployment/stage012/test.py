import boto3
import json

# 1. Initialize SageMaker runtime client
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")  # update region

# 2. Helper to call your endpoint
def classify_query(query_text: str):
    payload = json.dumps({"query": query_text})
    resp = runtime.invoke_endpoint(
        EndpointName="finance-to-sql-stage0-v4-hfmodel",
        ContentType="application/json",
        Body=payload
    )
    result = json.loads(resp["Body"].read().decode())
    return result  # e.g. [{"label":"LABEL_1","score":0.92}, ...]

# 3. Example usage
if __name__ == "__main__":
    sample_queries = [
        "What is the PAT?",
        "Compare ROE 2023 vs 2024",
        "Is there any anomaly in EBITDA?",
        "Forecast Revenue for next year"
    ]
    for q in sample_queries:
        print(q, "→", classify_query(q))
