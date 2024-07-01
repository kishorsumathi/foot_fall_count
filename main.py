from ultralytics import YOLO
import cv2
import numpy as np
import math

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("2.mp4")
assert cap.isOpened(), "Error opening video source"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

lines = []  # To store lines A and B
current_line = []

def draw_line(event, x, y, flags, param):
    global current_line, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_line) < 2:
            current_line.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(current_line) == 2:
            cv2.line(frame, current_line[0], current_line[1], (255, 0, 0), 2)
            lines.append((current_line[0], current_line[1]))
            current_line = []
        cv2.imshow("Frame (Press 'q' when done drawing lines)", frame)
        
def check_side_of_line(point, line):
    
    (x1, y1), (x2, y2) = line
    (x, y) = point
    result = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return np.sign(result)
# Read the first frame to allow line drawing
success, frame = cap.read()
if not success:
    raise Exception("Failed to read video")

cv2.imshow("Frame (Press 'q' when done drawing lines)", frame)
cv2.setMouseCallback("Frame (Press 'q' when done drawing lines)", draw_line)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

if len(lines) < 2:
    raise Exception("Two lines (A and B) must be drawn")

def distance_between_tuples(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


entries, exits = 0, 0
track_id_stat={}
previous_positions = {}
person_class_index=0
trajectories = {}

max_trajectory_length = 20 
while True:
    success, im0 = cap.read()
    if not success:
        break

    results = model.track(im0, persist=True, show=False, tracker="botsort.yaml")
    track_ids = results[0].boxes.id.int().cpu().tolist()

    for track_id, bbox,class_id in zip(track_ids, results[0].boxes.xyxy,results[0].boxes.cls):
        if int(class_id) == person_class_index:
            x1, y1, x2, y2 = map(int, bbox[:4])
            bbox=bbox.cpu().numpy()
            center_point = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if track_id not in trajectories:
                trajectories[track_id] = [center_point]
            else:
                trajectories[track_id].append(center_point)
                trajectories[track_id] = trajectories[track_id][-max_trajectory_length:]
            if len(trajectories[track_id]) > 1:
                cv2.polylines(im0, [np.array(trajectories[track_id], np.int32).reshape((-1, 1, 2))], False, (0, 0, 255), 2)

            # cv2.putText(im0, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            # print(center_point,lines[0])
            x1,y1=lines[0][0]
            x2,y2=lines[0][1]
            x3,y3=lines[1][0]
            x4,y4=lines[1][1]
            dist=distance_between_tuples((x2-x1/2,y2-y1/2),center_point)
            dist2=distance_between_tuples((x4-x3/2,y4-y3/2),center_point)
            print(dist,dist2)
            # dist=(x2-x1,y2-y1)-center_point
            # dist2=(x4-x3,y4-y3)-center_point
            if center_point>lines[0][0] and center_point<lines[0][1] and track_id not in track_id_stat.keys() and dist<dist2:
                exits+=1
                track_id_stat[track_id]="exits"
            elif center_point>lines[1][0] and center_point<lines[1][1] and track_id not in track_id_stat.keys() and dist>dist2:
                entries+=1
                track_id_stat[track_id]="entry"


    cv2.putText(im0, f'Entries: {entries}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(im0, f'Exits: {exits}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(im0, f'Total Peoples In Premises: {entries-exits}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 153, 153), 2)

    # Draw lines A and B
    for line in lines:
        cv2.line(im0, line[0], line[1], (255, 0, 0), 2)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', im0)
    out.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
