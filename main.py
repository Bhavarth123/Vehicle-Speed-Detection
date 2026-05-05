import cv2
import numpy as np
import time

net = cv2.dnn.readNet(
    r"E:/Sem 6/AI_p/yolov3-tiny.weights",
    r"E:/Sem 6/AI_p/yolov3-tiny.cfg"
)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("E:/Sem 6/AI_p/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

vehicle_classes = ["car", "bus", "truck", "motorbike"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(r"E:\Sem 6\AI_p\main_gate.mp4")

#cap.set(3, 960)
#cap.set(4, 540)

object_positions = {}
object_speeds = {}
object_static_frames = {}
next_object_id = 0

SPEED_LIMIT = 20

PIXEL_TO_METER_NEAR = 0.06
PIXEL_TO_METER_FAR = 0.02


def detect_vehicles(frame):
    global next_object_id, object_positions, object_speeds, object_static_frames

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320),
                                 (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    centers = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.25:
                label = classes[class_id]

                if label in vehicle_classes:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    centers.append((cx, cy))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    current_objects = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            current_objects.append((centers[i], x, y, w, h))

    updated_positions = {}

    for center, x, y, w, h in current_objects:
        matched_id = None

        for obj_id, prev_center in object_positions.items():
            if np.linalg.norm(np.array(center) - np.array(prev_center)) < 100:
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_object_id
            next_object_id += 1

        speed = object_speeds.get(matched_id, 0)

        if matched_id in object_positions:
            prev_center = object_positions[matched_id]

            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]

            pixel_distance = np.sqrt(dx**2 + dy**2)

            if pixel_distance < 10:
                object_static_frames[matched_id] = object_static_frames.get(matched_id, 0) + 1
            else:
                object_static_frames[matched_id] = 0

            if object_static_frames.get(matched_id, 0) > 10:
                continue

            y_ratio = center[1] / height
            scale = PIXEL_TO_METER_FAR + (PIXEL_TO_METER_NEAR - PIXEL_TO_METER_FAR) * y_ratio

            time_diff = 0.05

            if pixel_distance > 10:
                new_speed = ((pixel_distance * scale) / time_diff) * 3.6

                prev_speed = object_speeds.get(matched_id, 0)

                speed = (prev_speed * 0.85) + (new_speed * 0.15)

                if speed > 40:
                    speed = prev_speed

        object_speeds[matched_id] = speed
        updated_positions[matched_id] = center

        if speed < 5:
            continue

        
        color = (0, 255, 0)
        if speed > SPEED_LIMIT:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{int(speed)} km/h", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    object_positions = updated_positions

    return frame



while True:
    ret, frame = cap.read()

    if not ret:
        break

    
    frame = cv2.resize(frame, (960, 540))

    frame = detect_vehicles(frame)

    cv2.imshow("Speed Detection (Final Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()