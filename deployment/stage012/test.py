import boto3
import json

# 1. Initialize SageMaker runtime client
runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")  # update region

# 2. Helper to call your endpoint
def classify_query(query_text: str):
    input_data = {
        "query": query_text, 
        "taxonomy": "", 
        "currency": "", 
        "schema": "" , 
        "entity_id":"", 
        "scenario": "",
        "nature": ""
    }
    payload = json.dumps(input_data)
    resp = runtime.invoke_endpoint(
        EndpointName="v11-live-test",
        ContentType="application/json",
        Body=payload
    )
    result = json.loads(resp["Body"].read().decode())
    return result  # e.g. [{"label":"LABEL_1","score":0.92}, ...]

# 3. Example usage
if __name__ == "__main__":
    sample_queries = [
        "What is the net worth for July 2024",
        "What are the assets in june 2023",
        "What is the current ratio until June 2023",
        "Give me the net profit margin till July 2024"
        #"What is the PAT?", #label 0: Query
        #"Compare ROE 2023 vs 2024", #label 1: Comparison
        #"Is there any anomaly in EBITDA?", #label 2: Anomaly
        #"Forecast Revenue for next year" #label 3: Others
    ]
    for q in sample_queries:
        print(q, "→", classify_query(q))
        print('-----')
        print()
