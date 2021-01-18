import cv2
import sqlite3
import numpy as np

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

def insertOrUpdate(id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name="+str(Name)+" WHERE ID="+str(id)
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
id=raw_input('enter user id:')
name=raw_input('enter user name:')
insertOrUpdate(id,name)
sampleNum=0;
while(True):
    ret, img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sampleNum=sampleNum+1;
        cv2.imwrite("dataSet/user."+id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.waitKey(100);

    cv2.imshow('frame',img);
    cv2.waitKey(1);
    if(sampleNum>20):
        break

    
cam.release()
cv2.destroyAllWindows()
