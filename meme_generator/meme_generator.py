import cv2 as cv
import matplotlib.pyplot as plt
import streamlit as st

#loading the image and changing color and 
img = cv.imread('ressources/YOLO Resources/img/eagle.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

#copying the image
copy = img.copy()
font = cv.FONT_HERSHEY_SIMPLEX

# setting the 2 phrases taht should be added to the picture
text1 = 'This is not funny,'
text2 = 'but a Meme!'

# adding the phrases to the picture 
if len(text1) > 0 and len(text2) >0:
    textsize1 = cv.getTextSize(text1, font, 2, 2)
    textsize2 = cv.getTextSize(text2, font, 2, 2)
    textX1 = int((img_grey.shape[1] - textsize1[0][0]) / 2)
    textY1 = int(img.shape[0] - (img.shape[0] - 2 * textsize1[1]))
    textX2 = int((img_grey.shape[1] - textsize2[0][0]) / 2)
    textY2 = int((img.shape[0] - textsize2[1]))
    cv.putText(copy, text1, (textX1, textY1), font, 2, (0, 0, 0), 3)
    cv.putText(copy, text2, (textX2, textY2), font, 2, (0, 0, 0), 3)
    cv.imshow('Meme', copy)


# showing the Meme
plt.figure(figsize=(10, 8))
plt.imshow(copy)
plt.show()