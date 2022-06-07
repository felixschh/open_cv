import cv2
import numpy as np
import matplotlib.pyplot as plt

# import your handwritten numbers (picture)
pic = cv2.imread('hand_writing_recognition/handwritten_numbers.png')
grey = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
grey = cv2.resize(grey,(800,800))
plt.figure(figsize = (2,2))
cv2.imshow('grey_img', grey)
cv2.waitKey(0) 


# crop the image to show where the numbers are located
crop = pic[1500:2100,300:2300]
grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
cv2.imshow('crop', crop)


# transform image to get white letters and black background
ret, threshold = cv2.threshold(grey, 90, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('threshold', threshold)

# erosion and dilation
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

# dilate again
dilate = cv2.dilate(opening,kernel, iterations=15)
cv2.imshow('dilate', dilate)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# repeat
opening2 = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
dilate2 = cv2.dilate(opening2,kernel, iterations=15)

# getting the contours
contours, heirachy = cv2.findContours(dilate2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# drawing the contours around the numbers
image = crop.copy()
cv2.drawContours(image, contours,-1, (0,255,0),2)
cv2.imshow('contours', image)
