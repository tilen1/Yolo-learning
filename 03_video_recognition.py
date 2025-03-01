from ultralytics import YOLO 
import cv2
import cvzone
import math

def main():
    # chosing the yolo model
    model = YOLO('yolo_weights\\yolov8n.pt')

    # getting the video from the webcam
    capture = cv2.VideoCapture('videos\\cars.mp4')

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

    while True: 
        # getting the frame from video
        isTrue, frame = capture.read()

        # reading the frame from video
        result = model(frame, stream=True)

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

                # displaying all the data
                cvzone.putTextRect(frame, f'{conf} {class_names[object_id]}', (x1, max(10, y1 - 10)), scale=1, thickness=1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)

        # display the frame of the video with all the data
        cv2.imshow('Video', frame)
        # keep the frame for one second for it to be visable
        cv2.waitKey(1)

        # if we press d we stop video processing
        if cv2.waitKey(20) & 0xFF==ord('d'): 
            break

    capture.release()
    cv2.destroyAllWindows()


main()