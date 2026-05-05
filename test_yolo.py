import cv2
import time
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(r"D:\AI_p\main_gate_1.mp4")

WIDTH = 960
HEIGHT = 540
    
positions = {}
speeds = {}
static_frames = {}

SPEED_LIMIT = 10
MIN_MOVEMENT = 5       
MAX_SPEED = 40        
SMOOTH_FACTOR = 0.8    

while True:
    ret, frame = cap.read()
    if not ret:

        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

   
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is None:
        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0])

        
        if cls not in [2, 3, 5, 7]:
            continue

        track_id = int(box.id[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        speed = speeds.get(track_id, 0)

        if track_id in positions:
            prev_x, prev_y, prev_time = positions[track_id]

            dx = cx - prev_x
            dy = cy - prev_y
            dist = np.sqrt(dx**2 + dy**2)

            time_diff = time.time() - prev_time

           
            if dist < MIN_MOVEMENT:
                static_frames[track_id] = static_frames.get(track_id, 0) + 1
            else:
                static_frames[track_id] = 0

            if static_frames.get(track_id, 0) > 10:
                continue

   
            if time_diff > 0 and dist > MIN_MOVEMENT:
                scale = 0.025   

                new_speed = (dist * scale / time_diff) * 3.6

                prev_speed = speeds.get(track_id, 0)

                speed = (prev_speed * SMOOTH_FACTOR) + (new_speed * (1 - SMOOTH_FACTOR))

                if speed > MAX_SPEED:
                    speed = prev_speed

        speeds[track_id] = speed
        positions[track_id] = (cx, cy, time.time())

        if speed < 5:
            continue

        color = (0, 255, 0)
        if speed > SPEED_LIMIT:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"{int(speed)} km/h",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    cv2.imshow("Final Speed Detection (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



