import tensorflow as tf
import sagemaker
from sagemaker.tensorflow import TensorFlow
import boto3  
import os

def custom_fashion_recommendation_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # First Convolutional Layer with more filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Second Convolutional Layer with more filters
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        # Third Convolutional Layer with more filters
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),

        # Flatten the features for the fully connected layers
        tf.keras.layers.Flatten(),

        # First Fully Connected Layer with more units
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Second Fully Connected Layer with more units
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output Layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Replace 'your-bucket-name' with your actual S3 bucket name
bucket = 'fashionrepo'
s3_region = 'ap-northeast-2'  # Region where your S3 bucket is located
sagemaker_region = 'ap-northeast-2'  # Region where your SageMaker notebook instance is located

# Set the AWS credentials as environment variables
# ----
# os.environ['AWS_ACCESS_KEY_ID'] = 'ASIAXBDE6P26JQRJO2WN'
# os.environ['AWS_SECRET_ACCESS_KEY'] = '3ycRcI2bfMaGmbH0h8E30YE/jEUNj1hoXNF9d4Ys'
# os.environ['AWS_SESSION_TOKEN'] = 'IQoJb3JpZ2luX2VjEKD//////////wEaDmFwLW5vcnRoZWFzdC0yIkYwRAIgL1Yy+Oc3sbva1oPfBBXqdHy07VslxvwHBhM/2XjuMs8CIBM8zM0KRx3ZyuFrG9Z1Mq9KvJ2hjLtdUVB6hWXlURUiKvUCCDoQABoMNDgzMzk1NTM0NTI0IgyK7wOYwD2RVSiwct4q0gI9VzJp1ZdvpOTpx6DcacF/lZ3yhNq7qy8OpEs6NDB5Kylz+6AFKoKowgS592urcqq6uQ0t80GDWrITuJ2cbC4PTGPIf+f+K463EHDRHfxNAcbgVzTYI6xgcwclGPZreP1YzRN0CDPleDx/QpKPUAiHtWdVbPNcRgT7p84RfAc8APdF5StFX9QOfD79invze719GOWMp1YMcWi9J2c2gA5UwkTnEwIvgjctAFyeiePYMR3gDCBlMdJCC43wOxf408rcjKs4UOMJgaOFkBM5LOm9HS0e4Ess6k5BcoRbLob3Pzpi0SaprJHAnEvtq3LyFUBM5LzclQwVYYL5LcAAJaNui3x5G1iEDcJDBhZ5eg/c+M3tospb2kybbOicxY5oj2RBxq2olFek1n8vBdru3sPbzen3eysVjLmkViDPWHFk5KTBL9/JH3ADoWnjViGcuwAZjzDT0YGmBjqoAW05v6ZyKxJmlkOfaTgKzaECAC65e7tCXzlryuss7rN5W0Jw9gzdGWj4aqEVulF7NMBYHo31QGsOswe0V5sUmEnUnE+Hyf8NHNGWwV3NEGOGtlbnhMydizSBhX54rexYQy7SQiLGtBAfa3TkejqQDWf6NKTcmBqz8iKvuqFkHqDusKv417oIODdgkrnFVmosEGsQ+nlZUZwanUNkQXJfhxgCGGR3WRHTqQ=='
# ----
# Create a new AWS session with the necessary credentials and regions
# Use sagemaker_region while creating the session
session = boto3.Session(region_name=sagemaker_region)
sagemaker_session = sagemaker.Session(boto_session=session)

# Verify if the bucket exists in the specified region
s3_client = session.client('s3', region_name=s3_region)
bucket_exists = True
try:
    s3_client.head_bucket(Bucket=bucket)
except Exception as e:
    if e.response['Error']['Code'] == '404':
        bucket_exists = False

if not bucket_exists:

    raise ValueError(f"The S3 bucket '{bucket}' does not exist in the region '{s3_region}'.")

# Define the IAM role ARN for SageMaker
role = 'arn:aws:iam::483395534524:role/fashion_training_role'  # Replace with your IAM role ARN

# Create a TensorFlow Estimator with SageMaker
# estimator = TensorFlow(
#     entry_point='train.py',  # Your custom training script
#     role=role,
#     instance_count=1,
#     instance_type='ml.m4.xlarge',
#     framework_version='2.3.1',
#     py_version='py37',
#     output_path=f's3://{bucket}/output',
#     base_job_name='fashion-recommendation-training',
#     sagemaker_session=sagemaker_session,  # Pass the sagemaker_session parameter here
#     hyperparameters={
#         'input_shape': '[28, 28, 1]',
#         'num_classes': 10,
#         'epochs': 10,
#         'batch_size': 128,
#         'learning_rate': 0.001
#     }
# )
estimator = TensorFlow(
    entry_point='train.py',  # Your custom training script
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    framework_version='2.3.1',
    py_version='py37',
    output_path=f's3://{bucket}/output',
    base_job_name='fashion-recommendation-training',
    sagemaker_session=sagemaker_session,
    hyperparameters={
        'batch_size': 128,
        'epochs': 10,
        'learning_rate': 0.001,
        'input_shape': '[28, 28, 1]',  # Replace with your input shape
        'num_classes': 10,  # Replace with the number of classes in your dataset
    }
)

# Define the S3 URIs for 'train' and 'val' folders using the s3_region
train_data = f's3://{bucket}/train'
val_data = f's3://{bucket}/val'

# Train the model using SageMaker TensorFlow Estimator
estimator.fit({'train': train_data, 'validation': val_data})
