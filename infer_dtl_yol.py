import subprocess
import os
import re
from core.config.load_config import Config

# image_path = '/home/taymur/app_folder/backend_upd/static/uploads/arabic_celebrityframe_33.jpg'
# image_path = '/home/taymur/app_folder/backend_upd/chronos/dtl_yolov_2/impact.jpg'
person_model_path = Config.get_path('chronos/dtl_yolov_2/grip_yolov5s.pt')
club_model_path = Config.get_path('chronos/dtl_yolov_2/club_600_best.pt')

def detect(image_path, model_path):
    python_executable = Config.PYTHON_EXECUTABLE
    detect_script_path = Config.get_path('chronos/dtl_yolov_2/detect.py')

    result = subprocess.run(
        [python_executable, detect_script_path, '--weights', model_path, '--source', image_path, '--classes', '0'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print('Error:', result.stderr)
        return None

    output = result.stdout
    # print("Raw Detection Results:\n", output)

    # Updated regular expression pattern to extract class, confidence, and bbox
    pattern = r"Class: ([^\s]+) (\d\.\d+), BBox: \[(\d+), (\d+), (\d+), (\d+)\]"
    matches = re.findall(pattern, output)
    # print("Matches found:", matches)

    detections = []
    for match in matches:
        class_name, confidence, *bbox = match
        confidence = float(confidence)
        bbox = [int(x) for x in bbox]
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })

    return detections

def run_detection(image_path, person_model_path, club_model_path):
    person_detections = detect(image_path, person_model_path)
    club_detections = detect(image_path, club_model_path)
    # print('club_det',club_detections)
    # print('person_detections',person_detections)

    # Combine results from both models
    combined_detections = person_detections + club_detections

    # Dictionary to hold the highest confidence detection for each class
    highest_confidence_detections = {}

    for det in combined_detections:
        class_name = det['class']
        if class_name not in highest_confidence_detections or det['confidence'] > highest_confidence_detections[class_name]['confidence']:
            highest_confidence_detections[class_name] = det

    return list(highest_confidence_detections.values())



def is_within_range(combined_detections, threshold=0.10):
    person_bbox = None
    clubface_bbox = None

    # Extract person and clubface detections
    for det in combined_detections:
        if det['class'] == 'person':
            person_bbox = det['bbox']
        elif det['class'] == 'club':
            clubface_bbox = det['bbox']

    # Check if both detections are found
    if person_bbox and clubface_bbox:
        person_y2 = person_bbox[3]
        clubface_y2 = clubface_bbox[3]

        # Calculate 10% range
        lower_bound = person_y2 - (person_y2 * threshold)
        upper_bound = person_y2 + (person_y2 * threshold)

        # Check if clubface y2 is within the range
        return lower_bound <= clubface_y2 <= upper_bound

    return False

def find_detections(highest_conf_detections,class_name):
    for detection in highest_conf_detections:
        if detection['class'] == class_name:
            return detection['bbox']
    return None

def call_dtl_yolov(image_path):
    highest_conf_detections = run_detection(image_path, person_model_path, club_model_path)
    club_result_bool = is_within_range(highest_conf_detections)
    person_detections=find_detections(highest_conf_detections,'person')
    club_detections=find_detections(highest_conf_detections,'club')
    print('club_detections',club_detections)
    print('person_detections',person_detections)
    print('club_result_bool',club_result_bool)

    return club_result_bool,person_detections,club_detections

# call_dtl_yolov(image_path)
