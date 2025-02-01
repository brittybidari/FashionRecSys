# data_preparation.py - Updated

import pandas as pd
import os
import shutil
import boto3
from botocore.exceptions import NoCredentialsError
from sklearn.model_selection import train_test_split
import sys

def split_dataset_and_upload_to_s3(csv_file_path, root_directory, s3_bucket_name):
    try:
        # Load the CSV file containing the dataset information
        df = pd.read_csv(csv_file_path, usecols=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName'])
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return

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
        total_images = len(dataframe)
        skipped_images = 0

        for i, (index, row) in enumerate(dataframe.iterrows(), 1):
            image_filename = str(row['id']) + ".jpg"
            image_source_path = os.path.join(images_folder, image_filename)
            image_destination_path = os.path.join(destination_folder, image_filename)

            # Check if the image file exists in the destination folder
            if os.path.exists(image_destination_path):
                skipped_images += 1
                continue

            # Check if the image file exists in the source folder before copying
            if os.path.exists(image_source_path):
                shutil.copy(image_source_path, image_destination_path)
            else:
                print(f"Image file not found: {image_source_path}")

            # Calculate and display progress as a percentage
            sys.stdout.write(f"\rProgress: {i}/{total_images} ({i / total_images * 100:.2f}%)")
            sys.stdout.flush()

        # Add a new line after the progress display is complete
        print("\nImage copying completed!")
        print(f"Skipped {skipped_images} duplicate images.")

    # Copy images to 'train' and 'val' folders
    copy_images_to_folders(train_df, train_dir)
    copy_images_to_folders(val_df, val_dir)

    # Upload 'train' and 'val' folders to AWS S3 bucket
    def upload_to_s3(local_folder, s3_prefix):
        s3 = boto3.client('s3')
        total_files = sum(len(files) for _, _, files in os.walk(local_folder))
        uploaded_files = 0
        skipped_images = 0

        # Get the list of already uploaded files in S3
        s3_uploaded_files = set()
        try:
            response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
            for obj in response.get('Contents', []):
                s3_uploaded_files.add(obj['Key'])
        except Exception as e:
            print(f"Error listing objects in S3: {str(e)}")

        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                s3_path = os.path.join(s3_prefix, relative_path)

                # Check if the image file is already uploaded in S3
                if s3_path in s3_uploaded_files:
                    skipped_images += 1
                    continue

                try:
                    s3.upload_file(local_path, s3_bucket_name, s3_path)
                    uploaded_files += 1

                    # Calculate and display progress as a percentage
                    sys.stdout.write(f"\rProgress: {uploaded_files}/{total_files} ({uploaded_files / total_files * 100:.2f}%)")
                    sys.stdout.flush()

                except NoCredentialsError:
                    print("\nError: AWS credentials not found. Please check your AWS credentials.")
                    return
                except Exception as e:
                    print(f"\nError uploading {local_path} to S3 bucket: {str(e)}")
                    return

        # Add a new line after the progress display is complete
        print("\nUpload completed!")
        print(f"Skipped {skipped_images} duplicate images.")

    # Replace 'train' and 'val' with appropriate prefixes if desired
    train_prefix = 'train'
    val_prefix = 'val'

    # Upload 'train' and 'val' folders to the S3 bucket
    upload_to_s3(train_dir, train_prefix)
    upload_to_s3(val_dir, val_prefix)

# Usage:
csv_file_path = 'styles.csv'  # Replace with the path to the CSV file
root_directory = 'images'  # Replace with the root directory path
s3_bucket_name = 'image-recommendation-dataset'  # Replace with your S3 bucket name
split_dataset_and_upload_to_s3(csv_file_path, root_directory, s3_bucket_name)
