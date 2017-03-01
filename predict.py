import cv2
import glob
import random
import numpy as np


emotion_fishface = cv2.face.createFisherFaceRecognizer()

emotion_fishface.load('emotionfishface.xml')
picturename = raw_input("enter file name:")
print "COMMENCING FACIAL DETECTION AND EXPRESSION RECOGNITION"
image = cv2.imread(picturename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
img = cv2.resize(gray, (350, 350)) 
pred, conf = emotion_fishface.predict(img)
print pred
print conf
print "done"
pred = int(pred)
if pred == 0:
    print "neutral"
elif pred == 1:
    print "anger"
elif pred == 3:
    print "disgust"
elif pred == 5:
    print "happy"
elif pred == 7:
    print "surprise"


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
img = cv2.imread(picturename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    for (sx,sy,sw,sh) in smile_cascade.detectMultiScale(roi_gray):
        if pred == 5:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh),(0,0,255),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


