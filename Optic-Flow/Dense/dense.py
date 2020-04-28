import cv2
import numpy as np

cap = cv2.VideoCapture('chaplin.mp4')

ret, first_frame = cap.read()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

mask = np.uint8(first_frame)

mask[..., 1] = 255

while(cap.isOpened()):

    #Read capture and display the input frame
    ret, frame = cap.read()
    cv2.imshow('input',frame)
    
    #Convert all frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Calculate dense optical flow by Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
    
    #Computing magnitude and angle
    magn, angle = cv2.cartToPolar(flow[...,0],flow[...,1])
    
    #set the image hue according to optical flow direction
    mask[...,0] = angle*180/np.pi/2
    
    #Normalize the magnitude
    mask[...,2] = cv2.normalize(magn,None,0,255,cv2.NORM_MINMAX)
    
    #Convert HSV to RGB 
    rgb = cv2.cvtColor(mask,cv2.COLOR_HSV2RGB)
    
    #Display the new output and update previous frames
    cv2.imshow("DenseOpticalFLow",rgb)
    prev_gray=gray
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()