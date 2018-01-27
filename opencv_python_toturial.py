import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/home/wym/Downloads/google.png", 0)

cv2.imshow("google", img)
k = cv2.waitKey(0)
#for 64-bit machine, k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
	cv2.imwrite('googlegray.png', img)
	cv2.destroyAllWindows()


img = cv2.imread("/home/wym/Downloads/google.png", 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

#Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. So color
#images will not be displayed correctly in Matplotlib if image is read with Opencv2.
#
#Solution:
#b,g,r = cv2.split(img)
#img = cv2.merge([r, g, b])

#Capture Video from Camera
cap = cv2.VideoCapture(0)
#To capture a video, you need to create a VideoCapture object. Its argument can be either the device index or the name
#of a video file. Device index is just the number to specify which camera.

while True:
	# Capture frame-by-frame
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame', gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

#Playing Video from file
cap = cv2.VideoCapture('vtest.avi')
while(cap.isOpened()):
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

cap.release()
cv2.destroyAllWindows()
#Make sure proper versions of ffmpeg or gstreamer is installed. Sometimes, it is a headache to work with Video
#Capture mostly due to wrong installation of ffmpeg/gstreamer.

#Saving a Video
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#or fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
	    frame = cv2.flip(frame,0)

	    # write the flipped frame
	    out.write(frame)

	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	else:
	    break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

#1.2.3 Drawing Functions in OpenCV
#functions:cv2.line(), cv2.circle() , cv2.rectangle(), cv2.ellipse(), cv2.putText() etc.

#Drawing Line
#lineType;
#– LINE_8 (or omitted) - 8-connected line.
#– LINE_4 - 4-connected line.
#– LINE_AA - antialiased line.

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

#Drawing Rectangle
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

#Drawing Circle
img = cv2.circle(img,(447,63), 63, (0,0,255), -1)

#Drawing Ellipse
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

#Drawing Polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
#cv2.polylines() can be used to draw multiple lines. Just create a list of all the lines you want to draw
#and pass it to the function. All lines will be drawn individually. It is more better and faster way to draw a group of
#lines than calling cv2.line() for each line.

#Adding Text to Images
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

#1.2.4 Mouse as a Paint-Brush

#Simple Demo

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

#More Advanced Demo

drawing = False
# true if mouse is pressed

mode = True 
# if True, draw rectangle. Press 'm' to toggle to curve

ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
	global ix,iy,drawing,mode

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix,iy = x,y

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == True:
				cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
			else:
				cv2.circle(img,(x,y),5,(0,0,255),-1)

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == True:
			cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
		else:
			cv2.circle(img,(x,y),5,(0,0,255),-1)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
	cv2.imshow('image',img)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == 27:
		break

cv2.destroyAllWindows()
