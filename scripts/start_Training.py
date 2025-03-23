from sagemaker.estimator import Estimator
from sagemaker import get_execution_role

role = 'arn:aws:iam::241533125856:role/SageMakerExecutionRole-Interns'
print("about to start an instance ");
estimator = Estimator(
    image_uri='683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    use_spot_instances=True,
    max_run=1800,
    max_wait=3600,
    checkpoint_s3_uri='s3://vocasync/checkpoints/',
    output_path='s3://vocasync/model/',
    entry_point='train.py',
    source_dir='src',
    base_job_name='xgb-spot-job'
)


estimator.fit({'train': 's3://vocasync/data/train.csv'})

print("completed");
