# Drowsiness-Detection

In this model, we are finding whether the person is getting Drowsiness based on score we generated from CNN model. Model is trained on dataset having two classes, eyes are open or close. A pre-trained model is used for whether the feels sleepy or not.
It generating score based on prediction, score will increase with eyes closed and decreased with eyes open. If the model reaches score equal or above 15, it rings an alarm.
It is done with the help of OpenCV library. We are detecting face using Haar Cascade Model. Three Haar Cascadeb files are used for detecting face, left eye and right eye.
