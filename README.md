# Detection-code
#OpenCV: pip install opencv-python
#Pandas: pip install pandas
#NumPy: pip install numpy
#YOLO (Ultralytics): pip install ultralytics

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

def draw_text_with_background(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale=0.7, font_thickness=2, text_color=(255, 255, 255), 
                              bg_color=(0, 0, 0), padding=5, alpha=0.6):
    """Draw text with a semi-transparent background on an image."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position
    # Draw filled rectangle for the text background
    bg_rect = image[max(y-text_height-padding, 0):min(y+padding+baseline, image.shape[0]), 
                    max(x-padding, 0):min(x+text_width+padding, image.shape[1])]
    overlay = bg_rect.copy()
    cv2.rectangle(overlay, (0, 0), (bg_rect.shape[1], bg_rect.shape[0]), bg_color, -1)
    cv2.addWeighted(overlay, alpha, bg_rect, 1 - alpha, 0, bg_rect)
    # Put the text on top of the rectangle
    cv2.putText(image, text, (x, y + baseline), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Load rectangle points from file
rectangles = []
with open('rectangle_points.txt', 'r') as file:
    current_rectangle = []
    for line in file:
        if line.startswith("Rectangle"):
            if current_rectangle:
                rectangles.append(current_rectangle)
            current_rectangle = []
        else:
            x, y = map(int, line.strip().split(','))
            current_rectangle.append((x, y))
    if current_rectangle:
        rectangles.append(current_rectangle)

# Initialize video capture
cap = cv2.VideoCapture('vid4.MOV')

# Load class names from COCO dataset (full list)
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Load class names from object_classes.txt (filtered list, e.g., 'car')
with open("object_classes.txt", "r") as obj_file:
    obj_data = obj_file.read()
    object_classes = obj_data.split("\n")

# Find the indices of the filtered classes (in this case, 'car')
class_indices_to_detect = [class_list.index(obj_class) for obj_class in object_classes if obj_class in class_list]

# Initialize occupancy duration tracking variables
start_time_per_spot = [None] * len(rectangles)  # List to store the start time for each parking spot

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video when it ends
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame)
    detections = results[0].boxes.data
    df = pd.DataFrame(detections).astype("float")

    # Initialize occupancy status for each rectangle
    occupied_status = [False] * len(rectangles)

    # Iterate over all detected objects and filter only desired classes
    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        class_id = int(row[5])
        
        # Only process detections for classes in 'object_classes.txt'
        if class_id in class_indices_to_detect:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if the detection is inside any of the rectangles
            for rect_idx, rectangle in enumerate(rectangles):
                top_left, bottom_right = rectangle
                if top_left[0] <= cx <= bottom_right[0] and top_left[1] <= cy <= bottom_right[1]:
                    occupied_status[rect_idx] = True  # Mark the parking space as occupied
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle around the car
                    draw_text_with_background(frame, f'CAR {rect_idx + 1}', (x1, y1 - 20), font_scale=0.6, bg_color=(50, 50, 50))

    # Update occupancy duration tracking and display timer for each occupied spot
    for idx, status in enumerate(occupied_status):
        if status:
            if start_time_per_spot[idx] is None:  # If this is the first time the spot is occupied
                start_time_per_spot[idx] = time.time()  # Record the start time
            else:
                elapsed_time = time.time() - start_time_per_spot[idx]
                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
                draw_text_with_background(frame, f'Time: {elapsed_minutes}:{elapsed_seconds:02}', 
                                          (rectangles[idx][0][0], rectangles[idx][0][1] - 40), 
                                          font_scale=0.6, bg_color=(0, 0, 0))
        else:
            start_time_per_spot[idx] = None  # Reset the timer if the spot is no longer occupied

    # Output availability status for each parking spot
    for idx, status in enumerate(occupied_status):
        if status:
            print(f'Parking spot {idx + 1}: 0')  # 0 if occupied
        else:
            print(f'Parking spot {idx + 1}: 1')  # 1 if available

    # Draw all rectangles with unique numbers and color based on occupancy
    available_count = 0  # Initialize the count of available parking spaces
    for idx, rectangle in enumerate(rectangles):
        top_left, bottom_right = rectangle
        if occupied_status[idx]:
            color = (0, 0, 255)  # Red if occupied
        else:
            color = (0, 255, 0)  # Green if not occupied
            available_count += 1  # Increment available count
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        draw_text_with_background(frame, str(idx + 1), (top_left[0], top_left[1] - 10), font_scale=0.6, bg_color=(0, 0, 0))

    # Display the count of available parking spaces with a background box
    draw_text_with_background(frame, f'Available Parking Spaces: {available_count}', (10, 30), font_scale=1, bg_color=(50, 50, 50), alpha=0.8)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
        break

cap.release()
cv2.destroyAllWindows()
