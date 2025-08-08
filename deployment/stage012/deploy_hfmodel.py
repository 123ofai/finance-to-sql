# deploy.py
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.pytorch import PyTorchModel
from sagemaker import Session

# 1. IAM role (must have SageMaker permissions)
session = Session()
role_fin = "arn:aws:iam::641712484995:role/service-role/SageMaker-ExecutionRole-20250730T224347"

# 2. S3 path to your model.tar.gz (created by trainer.save_model + tar + upload)
model_fin = "s3://finalyzer-ai-ml-test/e2e/models_v2.tar.gz"

print('')
print('-------------')
hf_model = HuggingFaceModel(
    model_data           = model_fin,
    role                 = role_fin,
    image_uri = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04",
    #transformers_version = "4.26",
    #pytorch_version      = "1.13", #sagemaker doesn't allow any other version
    py_version           = "py312",
    entry_point          = "inference.py",
    source_dir           = "./src"  # where inference.py lives
)
print('Model created')

endpoint_name = "finance-to-sql-e2e-v11"
predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name=endpoint_name
)

print(f"Deployed endpoint: {predictor.endpoint_name}")
