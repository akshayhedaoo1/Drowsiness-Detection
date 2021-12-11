# Drowsiness-Detection

In this model, we are finding whether the person is getting Drowsiness based on score we generated from CNN model. Pre-trained Model is used which is trained on dataset having two classes, eyes are open or close.
It generating score based on prediction, score will increase with eyes closed and decreased with eyes open. If the model reaches score equal or above 15, it rings an alarm.
It is done with the help of OpenCV library. Detection of Face is done using Haar Cascade Model. Three Haar Cascade files are used for detecting face, left eye and right eye.


![drowsing](https://user-images.githubusercontent.com/84308415/145674989-657192e9-f400-4b6d-b455-9722846df392.gif)

