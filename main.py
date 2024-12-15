import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train4/weights/best.pt') 

cap = cv2.VideoCapture('pothole1.mp4')

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes  
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0]) 
            conf = box.conf[0]  
            cls = int(box.cls[0]) 
            label = f"Pothole {conf:.2f}"

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),  
                2,           
            )
            cv2.putText(
                frame,
                label,
                (x_min, y_min - 10),  
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,             
                (0, 255, 0),     
                2,              
                cv2.LINE_AA,
            )

    cv2.imshow('Pothole Detection (Press Q to Quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
