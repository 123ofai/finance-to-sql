import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
payload = json.dumps({"query": "profit margin for 2024"})
resp = runtime.invoke_endpoint(
    EndpointName="finance-to-sql-v1",
    ContentType="application/json",
    Body=payload
)
print(json.loads(resp["Body"].read()))