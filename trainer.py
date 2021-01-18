import os
import cv2
import numpy as np
from PIL  import Image

recognizer = cv2.createLBPHFaceRecognizer()
faceDetector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path='dataSet'

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L');
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split('.')[1])
        faces=faceDetector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
             faceSamples.append(imageNp[y:y+h,x:x+w])
             print Id
             Ids.append(Id)
        cv2.imshow("training",imageNp)
        cv2.waitKey(20)
    return Ids, faceSamples

Ids, faceSamples=getImagesAndLabels(path)
recognizer.train(faceSamples,np.array(Ids))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
