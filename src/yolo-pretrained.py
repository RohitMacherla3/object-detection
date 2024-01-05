from ultralytics import YOLO
import cv2

model = YOLO('../components/yolo-weights/yolov8n.pt')
results = model('/Users/rohitmacherla/Documents/Projects/object-detection/Images/bike-images.png', show=True)
cv2.waitKey(0)