import cv2 as cv
import numpy as np
from PIL import Image


img = Image.open('greenscreen/mc_background.png')
img.save('img.jpg')
img = cv.imread('img.jpg')


video = cv.VideoCapture('greenscreen/bateman_greenscreen.mp4')

w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
while True:
    retval, image = video.read()
    if retval:
        image = cv.resize(image, (w, h))
        img = cv.resize(img, (w, h))
        hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_green = (38, 40, 30)
        upper_green = (85, 255, 255)
        mask = cv.inRange(hsv_img, lower_green, upper_green)
        final = cv.bitwise_and(image, image, mask=mask)
        f = image - final
        f = np.where(f == 0, img, f)
        cv.imshow('video', f)
    else:
        video.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    if cv.waitKey(5) == ord('q'):
        break
video.release()
cv.destroyAllWindows()