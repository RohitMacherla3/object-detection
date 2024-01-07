from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import *

model = YOLO('src/components/yolo-weights/yolov8n.pt')
capture = cv2.VideoCapture('../Videos/people.mp4')

classNames = open('src/components/coco.names').read().strip().split('\n')

mask = cv2.imread('src/components/people-counter/mask.png')

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
people_count_up =[]
people_count_down = []

while True:
    suceess, img = capture.read()
    
    if not suceess:
        break
    
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    graphics = cv2.imread('src/components/people-counter/graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, graphics, (0,0))
    
    detections = np.empty((0, 5))
    
    for i in results:
        boxes = i.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            
            conf = math.ceil(box.conf[0]*100)/100
            
            cls_name = int(box.cls[0])
            cls_name = classNames[cls_name]
            
            if cls_name == 'person' and conf>0.3:
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))
        
    resultstracker = tracker.update(detections)  
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    for res in resultstracker:
        x1, y1, x2, y2, id = res
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10)
        cvzone.putTextRect(img, f'{id}', (max(0,x1), max(35, y1)), offset=5, scale=1, thickness=1)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limitsUp[0]< cx < limitsUp[2] and limitsUp[1]-50< cy < limitsUp[1]+50:
            if people_count_up.count(id)==0:
                people_count_up.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (255, 0, 255), 5)
                
        if limitsDown[0]< cx < limitsDown[2] and limitsDown[1]-50< cy < limitsDown[1]+50:
            if people_count_down.count(id)==0:
                people_count_down.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255, 0, 255), 5)
    
    cv2.putText(img, str(len(people_count_up)), (200, 90), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 8)
    cv2.putText(img, str(len(people_count_down)), (450, 90), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 8)
    cv2.imshow("Result", img)
    cv2.waitKey(1)
    


    
