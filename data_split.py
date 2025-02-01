import pandas as pd
import os
import shutil
import boto3
from botocore.exceptions import NoCredentialsError
from sklearn.model_selection import train_test_split

# Load the CSV file containing the dataset information
csv_file_path = 'styles.csv'  # Replace 'path/to/styles.csv' with the actual path to the CSV file
try:
    df = pd.read_csv(csv_file_path, usecols=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName'])
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    exit(1)

# Define the root directory where images will be organized into 'train' and 'val' folders
root_directory = 'images'  # Replace 'path/to/dataset' with the actual root directory path

# Create 'train' and 'val' directories
train_dir = os.path.join(root_directory, 'train')
val_dir = os.path.join(root_directory, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to copy images to the respective 'train' and 'val' folders
def copy_images_to_folders(dataframe, destination_folder):
    current_directory = os.getcwd()
    images_folder = os.path.join(current_directory, 'images')  # Assuming the 'images' folder is located in the current directory
    for index, row in dataframe.iterrows():
        image_filename = str(row['id']) + ".jpg"
        image_source_path = os.path.join(images_folder, image_filename)
        image_destination_path = os.path.join(destination_folder, image_filename)
        
        # Check if the image file exists before copying
        if os.path.exists(image_source_path):
            shutil.copy(image_source_path, image_destination_path)
        else:
            print(f"Image file not found: {image_source_path}")

# Copy images to 'train' and 'val' folders
copy_images_to_folders(train_df, train_dir)
copy_images_to_folders(val_df, val_dir)

# Upload 'train' and 'val' folders to AWS S3 bucket
def upload_to_s3(local_folder, s3_bucket, s3_prefix):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(s3_prefix, relative_path)
            try:
                s3.upload_file(local_path, s3_bucket, s3_path)
            except NoCredentialsError:
                print("Error: AWS credentials not found. Please check your AWS credentials.")
            except Exception as e:
                print(f"Error uploading {local_path} to S3 bucket: {str(e)}")

# Replace 'your-s3-bucket-name' with your actual S3 bucket name
s3_bucket_name = 'your-s3-bucket-name'
# Replace 'train' and 'val' with appropriate prefixes if desired
train_prefix = 'train'
val_prefix = 'val'

# Upload 'train' and 'val' folders to the S3 bucket
upload_to_s3(train_dir, s3_bucket_name, train_prefix)
upload_to_s3(val_dir, s3_bucket_name, val_prefix)
