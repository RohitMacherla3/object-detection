import streamlit as st
import numpy as np
from ultralytics import YOLO
from sort import *
import cv2
import cvzone
import math

def car_counter(model, input_video):
    capture = cv2.VideoCapture(input_video)
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    car_count = []
    stframes = st.empty()
    classNames = open('src/components/coco.names').read().strip().split('\n')
    
    while True:
        success, img = capture.read()
        if not success:
            break
        results = model(img, stream=True)
        imgGraphics = cv2.imread('src/components/car-counter/graphics.png', cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img, imgGraphics, (0,0))
        height, width, _ = img.shape
        limits = [300, height//2+200, width-200, (height//2)+200]
        detections = np.empty((0,5))
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                conf = math.ceil(box.conf[0]*100)/100
                cls_name = int(box.cls[0])
                cls_name = classNames[cls_name]
                
                if cls_name in ["car", "truck", "bus", "motorbike"] and conf>0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections=np.vstack((detections, current_array))
                    
        resulttracker = tracker.update(detections)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (1, 1, 1), 1)
        
        for res in resulttracker:
            x1, y1, x2, y2, id = res
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=10)
            cvzone.putTextRect(img, f'{id}', (max(0,x1), max(35, y1)), offset=5 ,  scale=0.5, thickness=1)
            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img, (cx, cy), 5, cv2.FILLED)
            
            if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
                if car_count.count(id)==0:
                    car_count.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                    
            
        cv2.putText(img, str(len(car_count)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,255), 8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframes.image(img_rgb, channels="RGB")
    return len(car_count)