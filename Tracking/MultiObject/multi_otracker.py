import cv2
import numpy as np 

## TO CHOOSE WEBCAM, LEAVE THE video_name=''
video_name = 'Vehicles.mp4'

#Defining tracker
def ask_for_tracker():


	
	print('Choose the tracker you want to use.')

    #Select the choice
	answer = input("Do you want help to choose the tracker type? (y/N): ")

	if answer=='y' or answer =='Y' or answer=='yes':

		print('Enter 0 for BOOSTING: Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost), but like Haar cascades, is over a decade old. This tracker is slow and doesn’t work very well. Interesting only for legacy reasons and comparing other algorithms ')
		print('Enter 1 for MIL: Better accuracy than BOOSTING tracker but does a poor job of reporting failure. ')
		print('Enter 2 for KCF: Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well. ')
		print('Enter 3 for TLD: TLD tracker was incredibly prone to false-positives')
		print('Enter 4 for MEDIANFLOW: Does a nice job reporting failures; however, if there is too large of a jump in motion the model will fail. ')
		print('Enter 5 for GOTURN: The only deep learning-based object detector included in OpenCV.')
		print('Enter 6 for MOSSE: Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed')
		print('Enter 7 for CSRT: Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower. ')

	else:
		print('Enter 0 for BOOSTING')
		print('Enter 1 for MIL')
		print('Enter 2 for KCF ')
		print('Enter 3 for TLD')
		print('Enter 4 for MEDIANFLOW ')
		print('Enter 5 for GOTURN ')
		print('Enter 6 for MOSSE ')
		print('Enter 7 for CSRT ')

	
	choice = input("Please select your tracker: ")

	if choice =='0':
		tracker = cv2.TrackerBoosting_create

	if choice =='1':
		tracker = cv2.TrackerMIL_create

	if choice =='2':
		tracker = cv2.TrackerKCF_create

	if choice =='3':
		tracker = cv2.TrackerTLD_create

	if choice =='4':
		tracker = cv2.TrackerMedianFlow_create

	if choice =='5':
		tracker = cv2.TrackerGOTURN_create

	if choice =='6':
		tracker = cv2.TrackerMOSSE_create

	if choice =='7':
		tracker = cv2.TrackerCSRT_create

	return tracker



tracker = ask_for_tracker()


tracker_name = str(tracker).split()[0][1:]


if video_name == '':
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture(video_name)

ret , frame = cap.read()

if not ret:
	print('Cannot read the video')

rects=[]
colors=[]


########### FOLLOW THE FOLLOWING INSTRUCTIONS TO SELECT ############################


print('PRINT ENTER OR SPACE TO CHOOSE THE NEXT OBJECT')
print('PRESS ENTER OR SPACE TWICE TO EXIT')


print('PLEASE IGNORE THE OTHER INSTRUCTIONS DURING SELECTION WHICH ARE SHOWN DEFAULT')

while(True):

	
	rect_box = cv2.selectROI("Selector",frame)
	colors.append(((np.random.randint(64,255)),(np.random.randint(64,255)),(np.random.randint(64,255))))

	key = cv2.waitKey(1) & 0xFF

	if rect_box==(0, 0, 0, 0):
		cv2.destroyWindow("Selector")
		break

	rects.append(rect_box)

multitracker = cv2.MultiTracker_create()

for rects_box in rects:
	multitracker.add(tracker(),frame, rects_box)


print("PRESS q TO EXIT THE VIDEO")


while(cap.isOpened()):
	success,frame = cap.read()

	if not success:
		break

	success,boxes = multitracker.update(frame)

	# To select an object to track in the middle of the video

	### PRESS 'S' TO SELECT AN OBJECT TO TRACK

	if cv2.waitKey(1) & 0xFF == ord('s'):
		box = cv2.selectROI(tracker_name,frame)
		multitracker.add(tracker(),frame,box)
		colors.append(((np.random.randint(64,255)),(np.random.randint(64,255)),(np.random.randint(64,255))))

	for i,new_box in enumerate(boxes):
		x,y,w,h = tuple(map(int,new_box))
		pts1 =(x,y)
		pts2 = (x+w,y+h)
		cv2.rectangle(frame,pts1,pts2,colors[i],2)
	
	cv2.imshow(tracker_name,frame)

	cv2.putText(frame,tracker_name,(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,125),3)

	#Exit with esc key
	if cv2.waitKey(27) & 0xFF == ord('q'):
		break

	cv2.waitKey(10)


cap.release()
cv2.destroyAllWindows()
