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
        
        #calculate base of ROI
        dest_roi = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        #meanshift to get coordinates of rectangle
        ret, track_window = cv2.meanShift(dest_roi,track_window,term_criteria)

        #draw rectangle on image
        x,y,w,h = track_window
        img = cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),3)
        cv2.imshow("FaceTracker",img)

        # Close the frame
        if cv2.waitKey(300) & 0xFF == ord("q"):
            break
    
# Release and Destroy
cap.release()
cv2.destroyAllWindows()