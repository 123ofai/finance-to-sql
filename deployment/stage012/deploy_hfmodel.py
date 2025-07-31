# deploy.py
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.pytorch import PyTorchModel
from sagemaker import Session

# 1. IAM role (must have SageMaker permissions)
session = Session()
role    = "arn:aws:iam::460493273466:role/AmazonSageMakerFullAccess-custom"

# 2. S3 path to your model.tar.gz (created by trainer.save_model + tar + upload)
model_uri = "s3://finalyser/finance-to-sql/v3/model.tar.gz"

print('')
print('-------------')
hf_model = HuggingFaceModel(
    model_data           = model_uri,
    role                 = role,
    transformers_version = "4.26",
    pytorch_version      = "1.13",
    py_version           = "py39",
    entry_point          = "inference.py",
    source_dir           = "./src"  # where inference.py lives
)
print('Model created')

endpoint_name = "finance-to-sql-e2e-v1-hfmodel"
predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name=endpoint_name
)

print(f"Deployed endpoint: {predictor.endpoint_name}")
