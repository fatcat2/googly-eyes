import cv2
import numpy as np

#import eyeball
imgEyeball = cv2.imread("eyeball.png")


#import erika
erika = cv2.resize(cv2.imread("evenmorefriends.jpg"), (0,0), fx=1, fy=1)

#import haar cascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')


gray = cv2.cvtColor(erika, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120), flags=cv2.CASCADE_SCALE_IMAGE)

#find faces in the image
for (x,y,w,h) in faces:
    cv2.rectangle(erika,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = erika[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray, minSize=(60, 60))
    for (ex,ey,ew,eh) in eyes:
    	eye = cv2.resize(imgEyeball, (ew, eh))
    	rows,cols,channels = eye.shape
    	erika_eye_roi = roi_color[ey:ey+cols, ex:ex+rows]
    	greyeye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    	ret, mask = cv2.threshold(greyeye, 10, 255, cv2.THRESH_BINARY)
    	mask_inv = cv2.bitwise_not(mask)
    	erika_bg = cv2.bitwise_and(erika_eye_roi, erika_eye_roi, mask = mask_inv)
    	eye_fg = cv2.bitwise_and(eye, eye, mask = mask)
    	dst = cv2.add(erika_bg, eye_fg)
        roi_color[ey:ey+ew, ex:ex+eh] = dst
        cv2.rectangle(roi_color,(ex, ey),(ex+ew,ey+eh),(0,255,0),2)
# small_img[0:20, 0:20] = cv2.resize(eye, (20, 20))
cv2.imshow("Eye Replacer", erika)
cv2.waitKey(0)
cv2.destroyAllWindows()
