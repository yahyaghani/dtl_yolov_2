import numpy as np
import tensorflow as tf
from PIL import Image

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array."""
    img = Image.open(path)
    img = img.resize((640, 640), Image.ANTIALIAS)
    return np.array(img)

def run_inference(model_path, image_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load image and preprocess
    image = load_image_into_numpy_array(image_path)
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        image = image / input_scale + input_zero_point

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the inference
    interpreter.invoke()

    # Extract the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def process_output(output_data, threshold=0.5):
    detections = []
    for detection in output_data[0]:
        # Assuming the format [ymin, xmin, ymax, xmax, confidence, class_prob]
        ymin, xmin, ymax, xmax, confidence, class_prob = detection
        if confidence > threshold:
            detections.append({
                "box": [xmin, ymin, xmax, ymax],
                "confidence": confidence,
                "class_prob": class_prob
            })
    return detections

def filter_detections(detections, confidence_threshold=0.5):
    filtered_detections = []
    for det in detections:
        if det['confidence'] > confidence_threshold:
            filtered_detections.append(det)
    return filtered_detections

# Example usage
model_path = 'grip_yolov5s-fp16.tflite'
image_path = 'captured_photo.jpg'
result = run_inference(model_path, image_path)
print(result)
processed_results = process_output(result)
filtered_results = filter_detections(processed_results, confidence_threshold=0.65)

for det in filtered_results:
    print(f"Box: {det['box']}, Confidence: {det['confidence']}, Class Probability: {det['class_prob']}")
