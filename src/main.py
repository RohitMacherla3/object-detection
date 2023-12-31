from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
results = model('/Users/rohitmacherla/Documents/Projects/object-detection/Images/school-images.png', show=True)
cv2.waitKey(0)