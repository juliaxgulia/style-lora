import boto3
import os

s3 = boto3.client('s3')
bucket = 'your-bucket-name'
local_folder = './data'

for filename in os.listdir(local_folder):
    if filename.endswith('.json') or filename.endswith('.png'):
        s3.upload_file(os.path.join(local_folder, filename), bucket, f'style-lora-data/{filename}')
