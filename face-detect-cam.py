import cv2
import time
face_detector = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml')

cam = cv2.VideoCapture('10_1.mp4')
# Lấy kích thước gốc của video
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original video resolution: {w}x{h}")

# Đặt cửa sổ hiển thị đúng tỷ lệ video
cv2.namedWindow('FRAME', cv2.WINDOW_NORMAL)
cv2.resizeWindow('FRAME', w, h)

count = 101
while True:
  OK, frame = cam.read()
  faces = face_detector.detectMultiScale(frame, 1.3, 5)
  time.sleep(0.5)
  for (x,y,w,h) in faces:
    if count > 200:
      break
    img_cut = cv2.resize(frame[y+3:y+h-3,x+3:x+w-3],(64,64))
    cv2.imwrite('img_10/p2_{}.png'.format(count),img_cut)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    count+=1
  cv2.imshow('FRAME',frame)
 
  if(cv2.waitKey(1) & 0xFF == ord('q')):
    break
  
cam.release()
cv2.destroyAllWindows()
