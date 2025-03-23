import pandas as pd
import boto3

# ==== Step 1: Generate Sample DataFrame ====
data = {
    'age': [25, 45, 35, 52, 29, 40],
    'income': [40000, 80000, 60000, 90000, 48000, 75000],
    'click_rate': [0.03, 0.12, 0.05, 0.09, 0.02, 0.11],
    'is_mobile': [1, 0, 1, 0, 1, 0],
    'target': [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
csv_filename = 'train.csv'
df.to_csv(csv_filename, index=False)

print(f"✅ CSV file '{csv_filename}' created successfully.")

# ==== Step 2: Upload to S3 ====
bucket_name = 'vocasync'
s3_key = 'data/train.csv'

# Initialize S3 client (uses default profile or SageMaker role)
s3 = boto3.client('s3')

try:
    s3.upload_file(csv_filename, bucket_name, s3_key)
    print(f"✅ Successfully uploaded '{csv_filename}' to 's3://{bucket_name}/{s3_key}'")
except Exception as e:
    print(f"❌ Failed to upload to S3: {e}")
