import os
import streamlit
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
import cv2

mixer.init()
sound = mixer.Sound(r'Drowsiness detection\alarm.wav')

face = cv2.CascadeClassifier("Drowsiness detection\haar cascade files\haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("Drowsiness detection\haar cascade files\haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("Drowsiness detection\haarcascade_righteye_2splits.xml")

label = ['open', 'close']
model = load_model("Drowsiness detection\models\cnnCat2.h5")
path = "Drowsiness detection"
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
rpred = [99]
lpred = [99]

while(True):
    ret, frame = cv2.VideoCapture(0).read()
    height, width = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, minNeighbors =5, scaleFactor =1.1, minSize = (25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 1)
    
    for (x,y, w,h) in left_eye:
        l_eye = frame[y:y+h,x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24,24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis = 0)
        lpred = model.predict(l_eye)
        if np.argmax(lpred, axis=1)[0] == 1:
            label = 'Open'
        elif np.argmax(lpred, axis=1)[0] == 0:
            label = 'Closed'
        break
    for (x,y, w,h) in right_eye:
        r_eye = frame[y:y+h,x:x+w]
        count = count+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24,24))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis = 0)
        rpred = model.predict(r_eye)
        if np.argmax(rpred, axis=1)[0] ==1:
            label = 'Open'
        elif np.argmax(rpred, axis=1)[0] ==0:
            label = 'Closed'
        break
    
    if (np.argmax(lpred, axis=1)[0] == 0 and np.argmax(rpred, axis=1)[0] == 0):
        score += 1
        cv2.putText(frame, 'Closed', (10, height-20), font, 1, (0, 0, 255), 1, cv2.LINE_8)
    else:
        score -= 1
        cv2.putText(frame, 'Open', (10, height-20), font, 1, (0, 255, 0), 1, cv2.LINE_8)
    
    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score :'  +str(score), (100, height-20), font, 1, (0, 0, 255), 1, cv2.LINE_8)
    
    if (score > 15):
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        cv2.rectangle(frame, (0,0), (width, height), (0,255, 255), 1)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.VideoCapture(0).release()
cv2.destroyAllWindows()
