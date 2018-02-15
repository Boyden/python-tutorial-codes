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

#Trackbar as the Color Palette

def nothing(x): 
    pass

# Create a black image, a window 
img = np.zeros((300,512,3), np.uint8) 
cv2.namedWindow('image')

# create trackbars for color change 
cv2.createTrackbar('R','image',0,255,nothing) 
cv2.createTrackbar('G','image',0,255,nothing) 
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality 
switch = '0 : OFF \n1 : ON' 
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1): 
    cv2.imshow('image',img) 
    k = cv2.waitKey(1) & 0xFF 
    if k == 27: 
        break
    # get current positions of four trackbars 
    r = cv2.getTrackbarPos('R','image') 
    g = cv2.getTrackbarPos('G','image') 
    b = cv2.getTrackbarPos('B','image') 
    s = cv2.getTrackbarPos(switch,'image')
    if s == 0: 
        img[:] = 0 
    else: 
        img[:] = [b,g,r]

cv2.destroyAllWindows()

#Basic Operations on Images
import cv2
import numpy as np

img = cv2.imread("C:\\Users\\cole\\Desktop\\logo.png")

shape = img.shape

img.item(10, 10, 2)
img.itemset((10, 10, 2), 100)

print(img.size)
print(img.dtype)

#Image ROI
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

#Splitting and Merging Image Channels
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

#or
b = img[:, :, 0]
img[:, :, 2] = 0
# cv2.split() is a costly operation (in terms of time), so only use it if necessary. Numpy indexing is much more efﬁcient and should be used if possible.

#Making Borders for Images (Padding)
import cv2
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255,0,0]

img1 = cv2.imread('C:\\Users\\cole\\Desktop\\logo.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()

#Arithmetic Operations on Images
#
#Image Addition
#There is a difference between OpenCV addition and Numpy addition. OpenCV addition is a saturated operation
#while Numpy addition is a modulo operation.
x, y = np.uint8([250]), np.uint8([10])

print(cv2.add(x, y))
print(x+y)

#Image Blending
#𝑑𝑠𝑡 = 𝛼 · 𝑖𝑚𝑔1 + 𝛽 · 𝑖𝑚𝑔2 + 𝛾
#dst(I) = saturate(src1(I) * alpha + src2(I) * beta + gamma)
img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.jpg')
dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Bitwise Operations
# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv_logo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Performance Measurement and Improvement Techniques
img1 = cv2.imread('messi5.jpg')

e1 = cv2.getTickCount()
for i in range(5,49,2):
    img1 = cv2.medianBlur(img1,i)
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print(t)

#Default Optimization in OpenCV
cv2.useOptimized()
%timeit res = cv2.medianBlur(img1,49)

cv2.setUseOptimized(False)
cv2.useOptimized()
%timeit res = cv2.medianBlur(img1,49)

cv2.setUseOptimized(True)

img = cv2.imread('opencv_logo.png')
#Measuring Performance in IPython
x = 5
%timeit y = x**2

%timeit y = x*x

z = np.uint8([5])
%timeit y = z*z

%timeit y = np.square(z)
#Python scalar operations are faster than Numpy scalar operations. So for operations including one or two
#elements, Python scalar is better than Numpy arrays. Numpy takes advantage when size of array is a little bit bigger.

%timeit z = cv2.countNonZero(img)

%timeit z = np.count_nonzero(img)
#Normally, OpenCV functions are faster than Numpy functions. So for same operation, OpenCV functions are
#preferred. But, there can be exceptions, especially when Numpy works with views instead of copies.

#There are several techniques and coding methods to exploit maximum performance of Python and Numpy. Only
#relevant ones are noted here and links are given to important sources. The main thing to be noted here is that, first try
#to implement the algorithm in a simple manner. Once it is working, profile it, find the bottlenecks and optimize them.
#
#1. Avoid using loops in Python as far as possible, especially double/triple loops etc. They are inherently slow.
#2. Vectorize the algorithm/code to the maximum possible extent because Numpy and OpenCV are optimized for
# vector operations.
#3. Exploit the cache coherence.
#4. Never make copies of array unless it is needed. Try to use views instead. Array copying is a costly operation.
#
#Even after doing all these operations, if your code is still slow, or use of large loops are inevitable, use additional
#libraries like Cython to make it faster.


#Image Processing in OpenCV
#Changing Colorspaces

import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

#Object Tracking
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


#How to find HSV values to track?
green = np.uint8([[[0,255,0]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print(hsv_green)
#Now you take [H-10, 100,100] and [H+10, 255, 255] as lower bound and upper bound respectively. Apart from this
#method, you can use any image editing tools like GIMP or any online converters to find these values, but don’t forget
#to adjust the HSV ranges.

#Image Thresholding
#Simple Thresholding
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gradient.png',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()


#Adaptive Thresholding
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dave.jpg',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

#Otsu’s Binarization
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('noisy2.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)', 'Original Noisy Image','Histogram',"Otsu's Thresholding", 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()


#How Otsu’s Binarization Works?
img = cv2.imread('noisy2.png',0)
blur = cv2.GaussianBlur(img,(5,5),0)

# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights

    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i
        
# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(thresh,ret)

#Geometric Transformations of Images

#Scaling
import cv2
import numpy as np

img = cv2.imread('messi5.jpg')
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#OR

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

#Translation
import cv2
import numpy as np

img = cv2.imread('messi5.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Warning: Third argument of the cv2.warpAffine() function is the size of the output image, which should be in
#the form of (width, height). Remember width = number of columns, and height = number of rows.

#Rotation
img = cv2.imread('messi5.jpg',0)
rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Affine Transformation

#In affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the
#transformation matrix, we need three points from input image and their corresponding locations in output image. Then
#cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.

img = cv2.imread('drawing.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')

plt.show()

#Perspective Transformation

#For perspective transformation, you need a 3x3 transformation matrix. Straight lines will remain straight even after
#the transformation. To find this transformation matrix, you need 4 points on the input image and corresponding points
#on the output image. Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be
#found by the function cv2.getPerspectiveTransform. Then apply cv2.warpPerspective with this 3x3 transformation
#matrix.

img = cv2.imread('sudokusmall.png')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

#Smoothing Images

#2D Convolution ( Image Filtering )
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

#Image Blurring (Image Smoothing)
#1. Averaging
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

#2. Gaussian Filtering
blur = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

#3. Median Filtering
def SaltAndPepper(src,percetage):

    NoiseImg=np.copy(src)
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    
    if len(src.shape) == 2:
    
        for i in range(NoiseNum):
            randX=random.randint(0,src.shape[0])
            randY=random.randint(0,src.shape[1])
            if random.randint(0,2)==0:
                NoiseImg[randX,randY]=0
            else:
                NoiseImg[randX,randY]=255   

        return NoiseImg 
    else:
        b, g, r = cv2.split(src)
        b, g, r = SaltAndPepper(b,percetage), SaltAndPepper(g,percetage), SaltAndPepper(r,percetage)
        NoiseImg = cv2.merge([b, g, r])
        return NoiseImg

noiseImg = SaltAndPepper(img, 0.1)
median = cv2.medianBlur(noiseImg,5)

plt.subplot(121),plt.imshow(noiseImg),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
        
#4. Bilateral Filtering

blur = cv2.bilateralFilter(img,9,75,75)

plt.subplot(121),plt.imshow(blur),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

#Morphological Transformations

#1. Erosion
import cv2
import numpy as np

img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(erosion),plt.title('Eroded')
plt.xticks([]), plt.yticks([])
plt.show()

#2. Dilation

dilation = cv2.dilate(img,kernel,iterations = 1)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dilation),plt.title('Dilated')
plt.xticks([]), plt.yticks([])
plt.show()

#3. Opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#4. Closing
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#5. Morphological Gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

#6. Top Hat
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

#7. Black Hat
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

#Opening operation:
#    dst = open(src; element) = dilate(erode(src; element))
#Closing operation:
#    dst = close(src; element) = erode(dilate(src; element))
#Morphological gradient:
#    dst = morph_grad(src; element) = dilate(src; element) - erode(src; element)
#“Top hat”:
#    dst = tophat(src; element) = src - open(src; element)
#“Black hat”:
#    dst = blackhat(src; element) = close(src; element) - src
#“Hit and Miss”: Only supported for CV_8UC1 binary images. Tutorial can be found in this page:
#https://web.archive.org/web/20160316070407/http://opencv-code.com/tutorials/hit-or-miss-transform-in-opencv/

#Structuring Element

# Rectangular Kernel
cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

#Image Gradients

#1. Sobel and Scharr Derivatives

#2. Laplacian Derivatives
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dave.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

#One Important Matter!
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('box.png',0)

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()

#Canny Edge Detection
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

#Image Pyramids
img = cv2.imread('messi5.jpg')
lower_reso = cv2.pyrDown(higher_reso)

higher_reso2 = cv2.pyrUp(lower_reso)

#Image Blending using Pyramids

#1. Load the two images of apple and orange
#2. Find the Gaussian Pyramids for apple and orange (in this particular example, number of levels is 6)
#3. From Gaussian Pyramids, find their Laplacian Pyramids
#4. Now join the left half of apple and right half of orange in each levels of Laplacian Pyramids
#5. Finally from this joint image pyramids, reconstruct the original image.

import cv2
import numpy as np,sys

A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:, 0:int(cols/2)], lb[:, int(cols/2):]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)


#Contours in OpenCV
import numpy as np
import cv2
imgray = cv2.imread('test.jpg')
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

cnt = contours[4]
img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

#1. Moments
import cv2
import numpy as np

img = cv2.imread('star.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
im, contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print(M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#2. Contour Area
area = cv2.contourArea(cnt)

#3. Contour Perimeter
perimeter = cv2.arcLength(cnt,True)

#4. Contour Approximation
