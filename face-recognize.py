import cv2
import numpy as np
import os
import tensorflow as tf
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D,Input

TRAIN_DATA = 'dataset/train_data'
TEST_DATA = 'dataset/test_data'

dict= {'CR7': [1,0], 'M10': [0,1]}
X_train = [] # Type of Xtrain [(matrix1, ohc1),(matrix2,ohc2),...]

def getData(path,lstData):
  for data in os.listdir(path):
    data_path = os.path.join(path,data) # Path to this folder data
    lst_img=[]
    for fileImg in os.listdir(data_path):
      fileImg_path = os.path.join(data_path, fileImg)
      label = fileImg_path.split('\\')
      img=cv2.imread(fileImg_path)
      lst_img.append((img,dict[label[1]]))
    lstData.extend(lst_img)
  return lstData


def makeModel(Xtrain):
  model = Sequential()
  model.add(Input(shape=(64, 64, 3)))
  model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.15))
  model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.2))
  model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(1000,activation='relu'))
  model.add(Dense(256,activation='relu'))
  model.add(Dense(2,activation='softmax'))

  #model.summary()

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  X = np.array([x[0] for x in X_train], dtype='float32') / 255.0
  y = np.array([x[1] for x in X_train], dtype='float32')

  model.fit(X,y,epochs=10)
  model.save('model-name-15epochs.h5')
  
def main():
  name=['Ronaldo','Messi']
  # X_train = getData(TRAIN_DATA, X_train) 
  # makeModel(X_train)
  face_detector = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml')
  myModel = load_model('model-name-15epochs.h5')
  print(type(myModel))
  cam = cv2.VideoCapture(0)
  while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
      roi = cv2.resize(frame[y:y+h,x:x+w],(64,64))
      result = np.argmax(myModel.predict(roi.reshape(-1,64,64,3)))
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
      cv2.putText(frame,name[result],(x+15,y-15),cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1)
    cv2.imshow('Frame',frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
      break
  

if __name__ == "__main__":
    main()