import boto3
import json

# 1. Initialize SageMaker runtime client
#runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")  # update region
runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")  # update region


# 2. Helper to call your endpoint
def classify_query(query_text: str):
    payload = json.dumps({"query": query_text})
    resp = runtime.invoke_endpoint(
        EndpointName="finance-to-sql-stage0-v1-hfmodel",
        ContentType="application/json",
        Body=payload
    )
    result = json.loads(resp["Body"].read().decode())
    return result  # e.g. [{"label":"LABEL_1","score":0.92}, ...]

# 3. Example usage
if __name__ == "__main__":
    sample_queries = [
        "What is the PAT?", #label 0: Query
        "Compare ROE 2023 vs 2024", #label 1: Comparison
        "Is there any anomaly in EBITDA?", #label 2: Anomaly
        "Forecast Revenue for next year" #label 3: Others
    ]
    for q in sample_queries:
        print(q, "→", classify_query(q))
