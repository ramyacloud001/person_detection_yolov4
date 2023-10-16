import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')
cap = cv2.VideoCapture('traffic_signs.mp4')  # 0 for default camera

person_in_roi = False

# Initialize variables for saving the video segment
out = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')
is_saving = False
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Define the region of interest (ROI)
    x1, y1, w1, h1 = 100, 150, 200, 200  
    roi = frame[y1:y1 + h1, x1:x1 + w1]
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Check if detected object is a person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Define ROI rectangle
                roi = frame[y:y+h, x:x+w]
                if person_in_roi:
    # If a person is in the ROI, start saving the video segment
                    if not is_saving:
                        out = cv2.VideoWriter("output_segment.avi", fourcc, 20.0, (w1, h1))
                        is_saving = True
                    out.write(roi)
                else:
                    if is_saving:
                        out.release()
                        is_saving = False

                # Draw a rectangle around the person
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), 2)
                cv2.imshow("Person Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
