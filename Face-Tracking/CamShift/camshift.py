import cv2
import numpy as np 

cap = cv2.VideoCapture('face_track.mp4')
ret,frame = cap.read()

face_casc = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
face_rects = face_casc.detectMultiScale(frame)

face_x,face_y, w, h = tuple(face_rects[0])
track_window = (face_x, face_y, w, h)

#setting up region of intrest(ROI) for tracking
roi = frame[face_y:face_y+h,face_x:face_x+w]

#HSV color mapping for ROI
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

#histogram for each from to calculate mean shift
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])

#Normalize the histogram
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX);

#setup termination criteria
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:

        #convert frame to hsv
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        #calculate ROI
        dest_roi = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        #camshift to get new coordinates for tracking window
        ret, track_window = cv2.CamShift(dest_roi,track_window,term_criteria)

        #Draw tracking window i.e rectangle on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img = cv2.polylines(frame,[pts],True,(0,255,0),5)

        #display
        cv2.imshow('Cam Shift',img)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
