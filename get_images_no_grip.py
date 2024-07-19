import csv
import os
import json
import requests
import re
from random import shuffle

# Define the labels and their indices
labels = {"person": 0, "club": 1, "grip": 2}

# Read the CSV file and shuffle it
rows = []
with open('dtl_lat.csv', 'r', newline='', encoding='utf-8') as csvfile:
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
split_ratio = 0.8
split_index = int(len(rows) * split_ratio)
train_rows = rows[:split_index]
valid_rows = rows[split_index:]

def extract_hash_code(filename):
    match = re.match(r"([a-fA-F0-9]{8})-", filename)
    return match.group(1) if match else None

def process_annotations(row, label_folder):
    hash_code = extract_hash_code(os.path.basename(row["image"]))

    if not hash_code:
        return

    annotations = json.loads(row["label"])
    with open(os.path.join(label_folder, f"{hash_code}.txt"), "w") as txtfile:
        for annotation in annotations:
            label_name = annotation["rectanglelabels"][0]
            if label_name == 'grip':
                continue

            x_center = (annotation["x"] + annotation["width"] / 2) / annotation["original_width"]
            y_center = (annotation["y"] + annotation["height"] / 2) / annotation["original_height"]
            width = annotation["width"] / annotation["original_width"]
            height = annotation["height"] / annotation["original_height"]
            label_index = labels[label_name]
            
            txtfile.write(f"{label_index} {x_center} {y_center} {width} {height}\n")

def organize_data(data, img_folder, label_folder):
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    for row in data:
        image_name = os.path.basename(row["image"])
        full_url = 'http://localhost:8080' + row["image"]
        response = requests.get(full_url, headers=headers)

        with open(os.path.join(img_folder, image_name), 'wb') as img_file:
            img_file.write(response.content)

        hash_code = extract_hash_code(image_name)

        if hash_code:
            new_image_name = hash_code + '.jpg'
            os.rename(os.path.join(img_folder, image_name), os.path.join(img_folder, new_image_name))
            process_annotations(row, label_folder)

organize_data(train_rows, "dataset2/train/images", "dataset2/train/labels")
organize_data(valid_rows, "dataset2/valid/images", "dataset2/valid/labels")
