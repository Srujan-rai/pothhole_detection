from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Use a pre-trained YOLO model (e.g., yolov8n.pt for nano variant)

# Train the model
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16)

# Validate the model
metrics = model.val()

# Save the best model
model.export(format='onnx')  # Export to ONNX for deployment
