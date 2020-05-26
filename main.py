import cv2
import numpy as np 
import yoloDetection
import dlibDetection
#from yoloDetection import yolo_detection

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    _, frame = cap.read()
    
    faces = dlibDetection.frontal_face(frame)
    print(faces)
    for face in faces:
        [x1, y1, x2, y2] = face
        cv2.rectangle(frame, (x1, y1), (x1+x2, y1+y2), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()