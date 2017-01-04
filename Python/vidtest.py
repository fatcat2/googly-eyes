import cv2
import numpy as np
# img = cv2.imread("moarfriends.jpg")
# small_img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
video = cv2.VideoCapture(0)
imgEyeball = cv2.imread("eyeball.png")
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
listfacenums = []
for x in range(100):
    ret, rev_frame = video.read()
    frame = cv2.flip(rev_frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)
    listfacenums.append(len(faces))
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye = cv2.resize(imgEyeball, (ew, eh))
            rows,cols,channels = eye.shape
            eye_roi = roi_color[ey:ey+cols, ex:ex+rows]
            greyeye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(greyeye, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            erika_bg = cv2.bitwise_and(eye_roi, eye_roi, mask = mask_inv)
            eye_fg = cv2.bitwise_and(eye, eye, mask = mask)
            dst = cv2.add(erika_bg, eye_fg)
            roi_color[ey:ey+ew, ex:ex+eh] = dst
    cv2.imshow("Eye Replacer", frame)
    cv2.waitKey(1)
total = 0
for x in listfacenums:
    total = x + total
print int(total / len(listfacenums))
video.release()
cv2.destroyAllWindows()
