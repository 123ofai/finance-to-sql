# deploy.py
import os
from sagemaker.pytorch import PyTorchModel
from sagemaker import Session
from sagemaker.async_inference import AsyncInferenceConfig


session = Session()
role    = "arn:aws:iam::460493273466:role/AmazonSageMakerFullAccess-custom"
bucket  = "finalyser"

model = PyTorchModel(
    model_data=f"s3://{bucket}/finance-to-sql/model.tar.gz",
    role=role,
    entry_point="e2e_sagemaker_nosql.py",
    source_dir="./code/",                # contains inference.py + requirements.txt
    framework_version="1.13",      # or your preferred PyTorch
    py_version="py39"
)
print('Model created')


endpoint_name = "finance-to-sql-v5-paramsret"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",  # choose based on your RAM needs
    endpoint_name=endpoint_name
)
"""

predictor = model.deploy(
    initial_instance_count=1,
    #instance_type="ml.m5.large",
    instance_type="ml.g5.xlarge",
    endpoint_name="async-endpoint-largecompute",
    async_inference_config=AsyncInferenceConfig(
        output_path="s3://finalyser/async_outputs/"
    )
)
"""

print("Deployed endpoint:", predictor.endpoint_name)
