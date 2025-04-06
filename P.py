import cv2

print("Package installed successfully")
face=cv2.CascadeClassifier("C:/Users/Himanshu/PycharmProjects/pythonProject/.venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
cap=cv2.VideoCapture(0)

cap.set(3,640)
while True:
   success,img=cap.read()
   col=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces=face.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
   for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("result",img)
   if cv2.waitKey(1) == ord("q"):
        break