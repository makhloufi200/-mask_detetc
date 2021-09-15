import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
# Load the cascade
face_cascade = cv2.CascadeClassifier('media/haarcascade_frontalface_alt2.xml')
model = load_model("test_video.h5")

# To capture video from webcam.
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

def convert_to_array(img):

    img = Image.fromarray(img, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def get_image_result(label):
    if label==0:

        return "With Mask"

    if label==1:

        return "Without Mask"


def predict_image(file):

    ar=convert_to_array(file)
    ar=ar/255
    #label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)

    label_index=np.argmax(score)

    acc=np.max(score)
    label=get_image_result(label_index)
    color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
    label = "{}: {:.2f}%".format(label, (acc) * 100)

    position = ((int) (img.shape[1]/2 - 268/2), (int) (img.shape[0]/2 - 36/2))
    cv2.putText(img, label, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
   
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        predict_image(img)

    cv2.imshow('Detect Mask Face With WebCam',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()