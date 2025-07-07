import cv2

## Path to image
img_path='img\p1.png'

## Include face detect FILTER
face_detector = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml')
## Read image
img=cv2.imread(img_path)

## Change image to black-white color
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

while True:
  faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
  count = 0
  for (x,y,w,h) in faces:
    img_cut = cv2.resize(img[y+3:y+h-3,x+3:x+w-3],(64,64))
    cv2.imwrite('img/p1_{}.png'.format(count),img_cut)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    count+=1
  ## Show image
  cv2.imshow('Frame',img)
  
  ## Quit image
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

