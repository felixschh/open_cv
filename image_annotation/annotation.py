import cv2 as cv


img = cv.imread('ressources/img/fry.jpg')

img_copy = img.copy()

# Hair
cv.rectangle(img_copy, (100,250), (400,25), (0,255,0))
cv.rectangle(img_copy, (100, 25), (200, 0), (0, 255, 0), cv.FILLED)
cv.putText(img_copy, 'Hair', (100, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Coat
cv.rectangle(img_copy, (100, 415), (370, 305), (0, 0, 255))
cv.rectangle(img_copy, (100, 305), (200, 280), (0, 0, 255), cv.FILLED)
cv.putText(img_copy, 'Coat', (100, 295), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Face
cv.rectangle(img_copy, (230, 330), (390, 175), (0, 255, 255))
cv.rectangle(img_copy, (230, 175), (330, 150), (0, 255, 255), cv.FILLED)
cv.putText(img_copy, 'Face', (230, 165), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


cv.imshow('image', img_copy)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)