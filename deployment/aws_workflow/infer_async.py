from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.session import Session
import json

session = Session()
predictor = Predictor(
    endpoint_name="async-endpoint",
    sagemaker_session=session,
    serializer=lambda x, y: json.dumps(x),
    deserializer=lambda b, c: json.loads(b.read())
)

async_predictor = AsyncPredictor(predictor)

# This will:
#  1) upload your JSON to a temp S3 path,
#  2) invoke the async endpoint,
#  3) poll for the output,
#  4) return the parsed result.
result = async_predictor.predict_async(
    data={"query":"profit in 2024"},
)
print(json.dumps(result, indent=2))
