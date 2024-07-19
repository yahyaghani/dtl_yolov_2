import torch
import torch.onnx

# Load your custom model
model = torch.load('club_f_best.pt', map_location=torch.device('cpu'))
model.eval()

# Create dummy input with the same shape as your model expects
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the shape to match your input

# Export the model
torch.onnx.export(model, dummy_input, "club_f_best.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])
