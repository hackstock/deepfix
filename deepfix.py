import numpy as np 
import cv2 
import dlib

weights_file = "./mmod_human_face_detector.dat"
# face_detector = dlib.cnn_face_detection_model_v1(weights_file)
face_detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(1)
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_detector(gray, 0)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x 
        h = face.bottom() - y 

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()