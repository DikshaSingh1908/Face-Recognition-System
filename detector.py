import cv2
import numpy as np
import sqlite3

faceDetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
recognizer=cv2.createLBPHFaceRecognizer(); 
recognizer.load("recognizer\\trainingdata.yml")
id=0
def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1,1 ) 
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetector.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[1]),(x,y+h+30),font,0);
            cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[2]),(x,y+h+60),font,0);
    cv2.imshow("frame",img);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cam.release()
cv2.destroyAllWindows()
