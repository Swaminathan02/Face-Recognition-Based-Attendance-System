import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread("known_faces/Raghu/2.jpg")  # change path as needed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print("Detected faces:", len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("Face Detection Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
