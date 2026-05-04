import cv2
import torch
import numpy as np
from yolov5 import YOLOv5


model = YOLOv5("yolov5s.pt", device="cpu")  


cap = cv2.VideoCapture("car_vedio.mp4")  

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1, 0) * max(0, y2 - y1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

  
    results = model.predict(frame)
    detections = results.pandas().xyxy[0]  

    accident_detected = False  
    vehicle_boxes = []  
    current_positions = {}

    for _, row in detections.iterrows():
        label = row["name"]
        conf = row["confidence"]
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])

       
        if label in ["car", "truck", "person"] and conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            if label in ["car", "truck"]:
                vehicle_boxes.append([x1, y1, x2, y2])
                current_positions[label] = (x1, y1)

           
            width, height = x2 - x1, y2 - y1
            if label == "person" and width > height * 1.5:
                accident_detected = True

  
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            iou = calculate_iou(vehicle_boxes[i], vehicle_boxes[j])
            if iou > 0.4: 
                accident_detected = True

    for label, pos in current_positions.items():
        if label in prev_positions:
            prev_x, prev_y = prev_positions[label]
            curr_x, curr_y = pos

            movement = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            if movement < 3:  
                accident_detected = True

    prev_positions = current_positions

  
    if accident_detected:
        cv2.putText(frame, "ACCIDENT DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print(" Accident Occurred! ")  

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()