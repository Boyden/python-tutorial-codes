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
