import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone

def web_cam_pred(model, web_video):
    web_video = cv2.VideoCapture(web_video)
    frame_placeholder = st.empty()
    stop_button = st.button('Stop')
    
    classNames = open('src/components/coco.names').read().strip().split('\n')
    
    while web_video.isOpened() and not stop_button:
        ret, frame = web_video.read()
        
        if not ret:
            st.write('Video capture stopped')
            break
        
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                cls_name = int(box.cls[0])
                cls_name = classNames[cls_name]
                cvzone.cornerRect(frame, (x1, y1, w, h), l=10)
                cvzone.putTextRect(frame, f'{cls_name}', (max(0,x1), max(35, y1)), offset=5 ,  scale=5, thickness=8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels= 'RGB')
        
        
        if cv2.waitKey(1) & 0xFF ==ord('q') or stop_button:
            break
    web_video.release()
    cv2.destroyAllWindows()