import torch
import numpy as np
import cv2
from PIL import Image
from models.yolo import Model  # Import the YOLOv5 model architecture
from core.config.load_config import Config


def load_model(model_path, model_config=None):
    if model_config is None:
        model_config = Config.get_path('chronos/dtl_yolov_2/models/yolov5s.yaml')
    
    # Create model architecture with the correct number of classes
    model = Model(model_config)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract the state dictionary
    state_dict = checkpoint['model'].state_dict() if 'model' in checkpoint else checkpoint

    # Load trained weights
    model.load_state_dict(state_dict)
    model.eval()
    return model
    
def detect(image_path, model):
    # Load image
    img = Image.open(image_path)
    img = np.array(img)

    # Convert color (PIL reads images in RGB, YOLOv5 expects BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Preprocess image
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # Convert to CHW
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        results = model(img)

    # Process results
    detections = []
    for det in results.xyxy[0]:  # results per image
        x1, y1, x2, y2, conf, cls = det
        detections.append({
            "class": int(cls),
            "confidence": float(conf),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

    return detections

# Load models
person_model_path = Config.get_path('chronos/dtl_yolov_2/grip_yolov5s.pt')
club_model_path = Config.get_path('chronos/dtl_yolov_2/club_f_best.pt')

person_model = load_model(person_model_path)
club_model = load_model(club_model_path)

# Example usage
image_path = Config.get_path('chronos/dtl_yolov_2/arabic_celebrityframe_33.jpg')
detections = detect(image_path, club_model_path)
print('detections',detections)
# ... process detections as needed
