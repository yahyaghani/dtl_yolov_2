import csv
import os
import json

# Define the labels and their indices
labels = {"person": 0, "club": 1, "grip": 2}

# Create a directory to store the txt files
os.makedirs("yolov5_annotations", exist_ok=True)

# Read the CSV file
with open('latest_dtl_yolo.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    
    for row in reader:
        # Extract the image name
        image_name = os.path.basename(row["image"]).split(".")[0]
        
        # Open a txt file to write the annotations
        with open(f"yolov5_annotations/{image_name}.txt", "w") as txtfile:
            
            # Load the JSON annotations
            annotations = json.loads(row["label"])
            
            for annotation in annotations:
                # Calculate normalized values
                x_center = annotation["x"] + annotation["width"] / 2
                x_center /= annotation["original_width"]
                
                y_center = annotation["y"] + annotation["height"] / 2
                y_center /= annotation["original_height"]
                
                width = annotation["width"] / annotation["original_width"]
                height = annotation["height"] / annotation["original_height"]
                
                label_index = labels[annotation["rectanglelabels"][0]]
                
                # Write to txt file
                txtfile.write(f"{label_index} {x_center} {y_center} {width} {height}\n")
