import cv2

def choose_tracker():
    print("-----------Tracker Menu-----------\n1.Boosting\n2.MIL\n3.KCF\n4.TLD\n5.Median Flow\n6.Goturn\n7.MOSSE\n8.CSRT")
    choice = input("Select any of the above tracker: ")

    if choice == '1':
        tracker = cv2.TrackerBoosting_create()
    if choice == '2':
        tracker = cv2.TrackerMIL_create()
    if choice == '3':
        tracker = cv2.TrackerKCF_create()
    if choice == '4':
        tracker = cv2.TrackerTLD_create()
    if choice == '5':
        tracker = cv2.TrackerMedianFlow_create()
    if choice == '6':
        tracker = cv2.TrackerGOTURN_create()
    if choice == '7':
        tracker = cv2.TrackerMOSSE_create()
    if choice == '8':
        tracker = cv2.TrackerCSRT_create()  
      
    return tracker

#call the choose_tracker function
tracker = choose_tracker()

tracker_name = str(tracker).split()[0][1:]

cap = cv2.VideoCapture('Vehicles.mp4')

ret,frame=cap.read()

roi = cv2.selectROI(frame)

#Initialise tracker

ret =tracker.init(frame,roi)

while True:
    ret, frame = cap.read()

    #update the frame by tracker  
    success,roi = tracker.update(frame)

    #convert roi from tuple to int
    (x,y,w,h) = tuple(map(int,roi))

    if success:
        pts1 = (x,y)
        pts2 = (x+w,y+h)
        cv2.rectangle(frame,pts1,pts2,(255,0,0),2)
    else:
        cv2.putText(frame,"Failed to Track",(100,200),cv2.FONT_HERSHEY_SIMPLEX,(255,255,0),3)
    
    #Display Tracker
    cv2.putText(frame,tracker_name,(20,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
    cv2.imshow(tracker_name,frame)

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()