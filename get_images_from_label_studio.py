import csv
import os
import json
import requests
import re  # Import regex module
from random import shuffle

# Define the labels and their indices
labels = {"person": 0, "club": 1, "grip": 2}

# Read the CSV file and shuffle it
rows = []
with open('dtl107.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        rows.append(row)
shuffle(rows)

# Your API Token
API_TOKEN = "4a0ba00c79fd5793691310d2968665f225813e07"
headers = {
    "Authorization": f"Token {API_TOKEN}"
}

# Split the data
split_ratio = 0.8  # 80% for training
split_index = int(len(rows) * split_ratio)
train_rows = rows[:split_index]
valid_rows = rows[split_index:]

# Function to extract hash code from filename
def extract_hash_code(filename):
    match = re.match(r"([a-fA-F0-9]{8})-", filename)
    return match.group(1) if match else None

# Function to organize images and labels
def organize_data(data, img_folder, label_folder):
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    for row in data:
        # Handle images
        print('row', row)
        image_name = os.path.basename(row["image"])
        full_url = 'http://localhost:8082' + row["image"]
        response = requests.get(full_url, headers=headers)
        print('full url', full_url)
        print('response', response)

        with open(os.path.join(img_folder, image_name), 'wb') as img_file:
            img_file.write(response.content)

        # Extract hash code from image filename
        hash_code = extract_hash_code(image_name)

        if hash_code:
            # Rename image based on hash code
            new_image_name = hash_code + '.jpg'
            os.rename(os.path.join(img_folder, image_name), os.path.join(img_folder, new_image_name))
            
            # Handle labels
            for label_file in os.listdir('./yolov5_annotations/'):
                if extract_hash_code(label_file) == hash_code:
                    label_path_src = os.path.join('./yolov5_annotations/', label_file)
                    label_path_dst = os.path.join(label_folder, hash_code + '.txt')
                    os.rename(label_path_src, label_path_dst)
                    break

# Organize train and valid datasets
organize_data(train_rows, "dataset/train/images", "dataset/train/labels")
organize_data(valid_rows, "dataset/valid/images", "dataset/valid/labels")
