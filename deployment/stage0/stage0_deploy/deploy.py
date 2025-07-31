# deploy.py
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.pytorch import PyTorchModel
from sagemaker import Session

# 1. IAM role (must have SageMaker permissions)
session = Session()
role    = "arn:aws:iam::460493273466:role/AmazonSageMakerFullAccess-custom"

# 2. S3 path to your model.tar.gz (created by trainer.save_model + tar + upload)
model_uri = "s3://finalyser/finance-to-sql/v2/model.tar.gz"

# 3. Create the HuggingFaceModel
"""
hf_model = HuggingFaceModel(
    model_data           = model_uri,
    role                 = role,
    transformers_version = "4.26",     # match your training stack
    pytorch_version      = "1.13",     
    py_version           = "py39",
    entry_point          = "inference.py",  # your handler above
    # optionally, specify source_dir if you bundle requirements.txt here
    # source_dir         = "stage0_deploy/" 
)
"""
model = PyTorchModel(
    model_data=model_uri,
    role=role,
    entry_point="inference.py",
    source_dir="stage0_deploy/",                # contains inference.py + requirements.txt
    framework_version="1.13",      # or your preferred PyTorch
    py_version="py39"
)
print('Model created')


endpoint_name = "finance-to-sql-stage0-v1"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",  # choose based on your RAM needs
    endpoint_name=endpoint_name
)

"""
# 4. Deploy to an endpoint
predictor = hf_model.deploy(
    initial_instance_count = 1,
    instance_type          = "ml.m5.large",
    endpoint_name          = "stage0-classifier-endpoint"
)
"""

print(f"Deployed endpoint: {predictor.endpoint_name}")
