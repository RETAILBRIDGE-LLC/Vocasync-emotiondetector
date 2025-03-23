import boto3
from sagemaker import Session
from sagemaker.model import Model
import time

# === CONFIGURATION ===
role = 'arn:aws:iam::241533125856:role/SageMakerExecutionRole-Interns'
region = 'us-east-2'
model_s3_path = 's3://vocasync/model/output/model.tar.gz'  # <-- Adjust path if different
image_uri = '683313688378.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.3-1'
endpoint_name = f"xgb-endpoint-{int(time.time())}"  # unique name with timestamp

# === SETUP SESSION ===
session = Session(boto3.Session(region_name=region))

# === STEP 1: Create SageMaker Model ===
model = Model(
    image_uri=image_uri,
    model_data=model_s3_path,
    role=role,
    sagemaker_session=session
)

# === STEP 2: Deploy Endpoint ===
print(f"🚀 Deploying model to endpoint: {endpoint_name}...")
predictor = model.deploy(
    instance_type='ml.t2.medium',
    initial_instance_count=1,
    endpoint_name=endpoint_name
)

print("✅ Endpoint is live!")

# === STEP 3: Send Test Prompt ===
test_input = "40,70000,0.08,1"  # Example: age, income, click_rate, is_mobile
print(f"🧪 Sending test input: {test_input}")
response = predictor.predict(test_input)
print("🧠 Prediction result:", response)

# === STEP 4: Clean Up ===
delete = input("\n❓ Do you want to delete the endpoint now? (y/n): ").lower()
if delete == 'y':
    print(f"🧹 Deleting endpoint: {endpoint_name}")
    predictor.delete_endpoint()
    print("✅ Endpoint deleted.")
else:
    print(f"⚠️ Endpoint '{endpoint_name}' still running. Remember to delete it later to avoid charges.")
