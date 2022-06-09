import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import figure


# plt.figure(figsize=(20,15))
# green_screen = cv.imread('/Users/felixschekerka/Desktop/computer_vision/greenscreen.png')
background = cv.imread('greenscreen/mc_background.png')
background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
# dog = cv.imread('/Users/felixschekerka/Desktop/computer_vision/Computer Vision Course Resources/img/dog.jpg')
# dog = cv.cvtColor(dog, cv.COLOR_BGR2RGB)
mc_gs = cv.imread('greenscreen/minecraft_greenscreen.png')
mc_gs = cv.cvtColor(mc_gs, cv.COLOR_BGR2RGB)

lower_green = np.array([0,100,0])
upper_green = np.array([120, 255, 100])

mask = cv.inRange(mc_gs, lower_green, upper_green)
# plt.imshow(mask, cmap='gray')

masked_image = np.copy(mc_gs)
masked_image[mask != 0] = [0,0,0] # setting background to black

background_edit = background[0:370, 0:700]
background_edit[mask == 0] = [0, 0, 0]
background_edit[mask == 0] = [0,0,0]

merged_image = background_edit + masked_image #adding both arrays works like merging the 2 pictures

plt.figure(figsize=(10, 8))
plt.imshow(merged_image)
plt.show()