import csv
import os
import json
import re

# Define the labels and their indices
labels = {"person": 0, "club": 1, "grip": 2}

# Create a directory to store the txt files
os.makedirs("yolov5_annotations", exist_ok=True)

# Function to extract hash code from filename
def extract_hash_code(filename):
    match = re.match(r"([a-fA-F0-9]{8})-", filename)
    return match.group(1) if match else None

# Read the CSV file
with open('latest_dtl_yolo.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    
    for row in reader:
        # Extract the image name and hash code
        image_name = os.path.basename(row["image"])
        hash_code = extract_hash_code(image_name)

        # Use hash code as the base name for the output text file, skip if not found
        if not hash_code:
            continue

        # Open a txt file to write the annotations
        with open(f"yolov5_annotations/{hash_code}.txt", "w") as txtfile:
            
            # Load the JSON annotations
            annotations = json.loads(row["label"])
            
            for annotation in annotations:
                label_name = annotation["rectanglelabels"][0]

                # Skip writing if the label is 'grip'
                if label_name == 'grip':
                    continue

                # Calculate normalized values
                x_center = annotation["x"] + annotation["width"] / 2
                x_center /= annotation["original_width"]
                
                y_center = annotation["y"] + annotation["height"] / 2
                y_center /= annotation["original_height"]
                
                width = annotation["width"] / annotation["original_width"]
                height = annotation["height"] / annotation["original_height"]
                
                label_index = labels[label_name]
                
                # Write to txt file
                txtfile.write(f"{label_index} {x_center} {y_center} {width} {height}\n")
