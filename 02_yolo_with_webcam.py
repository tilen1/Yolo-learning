from ultralytics import YOLO 
import cv2 as cv 

def main():
    capture = cv.VideoCapture(0)
    change_res(capture, 640, 480)

    model = YOLO('yolo_weights\\yolov8n.pt')

    while True: 
        isTrue, frame = capture.read()
        results = model(frame, stream=True)

        for res in results: 
            boxes = res.boxes
            print(boxes)
            for box in boxes: 
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)

        cv.imshow('Live video', frame)

        if cv.waitKey(20) & 0xFF==ord('d'): 
            break 

    capture.release()
    cv.destroyAllWindows()


def change_res(capture, width, height):
    capture.set(3, width)
    capture.set(4, height)


main()