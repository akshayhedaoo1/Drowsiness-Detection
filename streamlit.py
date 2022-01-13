
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from tensorflow.keras.models import load_model
import tensorflow as tf 
import os
from pygame import mixer
import cv2
import numpy as np
import streamlit as st



class VideoTransformer(VideoTransformerBase):

    def __init__(self):
        
        self.face = cv2.CascadeClassifier(r'/home/akshayhedaoo/Desktop/Drowsiness_Detection/haar cascade files/haarcascade_frontalface_alt.xml')
        self.leye = cv2.CascadeClassifier(r'/home/akshayhedaoo/Desktop/Drowsiness_Detection/haar cascade files/haarcascade_lefteye_2splits.xml')
        self.reye = cv2.CascadeClassifier(r'/home/akshayhedaoo/Desktop/Drowsiness_Detection/haar cascade files/haarcascade_righteye_2splits.xml')

        self.model = load_model(r'/home/akshayhedaoo/Desktop/Drowsiness_Detection/Model.h5')
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.path = r'/home/akshayhedaoo/Desktop/Drowsiness_Detection'
        self.score = 0
    
    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        height, width = frame.shape[:2]
        
        mixer.init()
        sound = mixer.Sound('/home/akshayhedaoo/Desktop/Drowsiness_Detection/alarm.wav')


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, minNeighbors =5, scaleFactor =1.1)
        l_eye = self.leye.detectMultiScale(gray)
        r_eye = self.reye.detectMultiScale(gray)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (128, 128, 128), 1)
    
        for (x,y, w,h) in l_eye:
            l_eye = frame[y:y+h,x:x+w]
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
            l_eye = cv2.resize(l_eye, (32, 32))
            l_eye = np.expand_dims(l_eye, axis = 0)
            lpred = self.model.predict(l_eye)
        
        for (x,y, w,h) in r_eye:
            r_eye = frame[y:y+h,x:x+w]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
            r_eye = cv2.resize(r_eye, (32, 32))
            r_eye = np.expand_dims(r_eye, axis = 0)
            rpred = self.model.predict(r_eye)
        
        if (np.argmax(lpred, axis =1)[0] == 0 and np.argmax(rpred, axis =1)[0] == 0):
            self.score += 1
            cv2.putText(frame, 'Closed', (10, height), self.font, 1, (0, 0, 255), 1, cv2.LINE_8)
        else:
            self.score -= 1
            cv2.putText(frame, 'Open', (10, height), self.font, 1, (0, 255, 0), 1, cv2.LINE_8)

        if (self.score < 0):
            self.score = 0
        cv2.putText(frame, 'Score :'  +str(self.score), (100, height), self.font, 1, (0, 0, 255), 1, cv2.LINE_8)       
        if (self.score > 15):
            cv2.imwrite(os.path.join(self.path, 'image.jpg'), frame)
            try:
                sound.play()
                if self.score < 15:
                    mixer.pause()
                else:
                    mixer.resume()
            except:
                pass
            cv2.rectangle(frame, (0,0), (width, height), (0,0, 255), 5)
        return frame


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)