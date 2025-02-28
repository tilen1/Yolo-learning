from ultralytics import YOLO 
import cv2 as cv 

model = YOLO('yolo_weights\\yolov8n.pt')
results = model('C:\\Users\\tilen\\Desktop\\PycharmProjects\\pythonProject\\image_recognition\\yolo\\images\\cars.jpg', show=True)

cv.waitKey(0)