from ultralytics import YOLO 
import cv2
import cvzone
import math
import sys
sys.path.append('tracker')
from sort import *

def main():
    # chosing the yolo model
    model = YOLO('yolo_weights\\yolov8l.pt')

    # getting the video from the webcam
    capture = cv2.VideoCapture('videos\\people.mp4')

    # craating a mask -> from canva.com
    mask = cv2.imread('images\\mask_escalator.jpg')

    # class of all detectable objects
    class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
    
    # creating a tracker
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

    # counted objects 
    object_count_up = 0
    counted_objects_up = []
    object_count_down = 0
    counted_objects_down = []

    # coordinates of the tracking lines
    line_coordinates_up = [(140, 300), (430, 300)]
    line_coordinates_down = [(430, 300), (670, 300)]

    while True: 
        # getting the frame from video
        isTrue, frame = capture.read()
        
        # creating a masked image
        masked_frame = cv2.bitwise_and(frame, mask)

        # reading the frame from video
        result = model(masked_frame, stream=True)

        # creating a array for tracker data (x1, y1, x2, y2, conf)
        detections = np.empty((0,5))

        # yolo model can read multiple images at once so we have to iterate over them 
        # even if we only have one image
        for im in result: 
            # read the results of each image and store it into a list boxes
            boxes = im.boxes

            for box in boxes: 
                # this is still a (1,4) tensor (list in a list) so we have to unpack it even though 
                # there is only a single object in this tensor 
                # getting the data for the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # getting a confidence of reading from over detected object
                conf = math.ceil(box.conf[0]*100) / 100

                # getting a class index for each reading
                object_id = int(box.cls[0])


                if class_names[object_id] in ['person'] and conf > 0.4:
                    # displaying all the data
                    cvzone.putTextRect(frame, f'{conf} {class_names[object_id]}', (x1, max(10, y1 - 10)), scale=1, thickness=1, offset=2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

                    # creating the data array for the detected object
                    object_array = np.array([x1, y1, x2, y2, conf])
                    # putting the data in the detections array
                    detections = np.vstack((detections, object_array))


        # getting the results from the tracker
        results = tracker.update(detections)

        # drawing the line detection line
        # cv2.line(frame, line_coordinates[0], line_coordinates[1], (0, 0, 255), 5)  # remove later
        cv2.line(frame, line_coordinates_up[0], line_coordinates_up[1], (0, 255, 0), 5)
        cv2.line(frame, line_coordinates_down[0], line_coordinates_down[1], (0, 0, 255), 5)

        for result in results: 
            # unpacking the data from the tracker
            x1, y1, x2, y2, object_id = result
            x1, y1, x2, y2, object_id = int(x1), int(y1), int(x2), int(y2), int(object_id)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # drawing the data from tracker
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cvzone.putTextRect(frame, f'{object_id}', (x1, max(10, y1 - 25)), scale=1, thickness=1, colorR=(255,0,0), offset=2)

            # counting the objects that crossed the line and went up 
            if line_coordinates_up[0][0]  < cx < line_coordinates_up[1][0] and line_coordinates_up[0][1] - 20 < cy < line_coordinates_up[0][1] + 20 and object_id not in counted_objects_up:
                object_count_up += 1
                counted_objects_up.append(object_id)
                cv2.line(frame, line_coordinates_up[0], line_coordinates_up[1], (255, 255, 255), 20)

            # counting the objects that crossed the line and went down
            if line_coordinates_down[0][0]  < cx < line_coordinates_down[1][0] and line_coordinates_down[0][1] - 20 < cy < line_coordinates_down[0][1] + 20 and object_id not in counted_objects_down:
                object_count_down += 1
                counted_objects_down.append(object_id)
                cv2.line(frame, line_coordinates_down[0], line_coordinates_down[1], (255, 255, 255), 20)

            # drawing the line over couned objects
            if object_id in counted_objects_up or object_id in counted_objects_down:
                cv2.line(frame, (x1, cy), (x2, cy), (0, 255, 0), 4)

        # draw object count 
        cvzone.putTextRect(frame, f'counted up:{str(object_count_up)}', (800, 360), scale=3, thickness=3, colorR=(0, 255, 0))
        cvzone.putTextRect(frame, f'counted down:{str(object_count_down)}', (800, 420), scale=3, thickness=3, colorR=(0, 0, 255))

        # display the frame of the video with all the data
        cv2.imshow('Video', frame)
        # keep the frame for one second for it to be visable
        cv2.waitKey(10)

        # if we press d we stop video processing
        if cv2.waitKey(1) & 0xFF==ord('d'): 
            break

    capture.release()
    cv2.destroyAllWindows()


main()