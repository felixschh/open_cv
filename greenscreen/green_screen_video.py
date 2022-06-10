import cv2
import numpy as np
from PIL import Image


img = Image.open('greenscreen/mc_background.png')
img.save('img.jpg')
img = cv2.imread('img.jpg')


video = cv2.VideoCapture('greenscreen/bateman_greenscreen.mp4')

w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    retval, image = video.read()
    if retval:
        image = cv2.resize(image, (w, h))
        img = cv2.resize(img, (w, h))
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = (38, 40, 30)
        upper_green = (85, 255, 255)
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
        final = cv2.bitwise_and(image, image, mask=mask)
        f = image - final
        f = np.where(f == 0, img, f)
        cv2.imshow('video', f)
    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if cv2.waitKey(5) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()