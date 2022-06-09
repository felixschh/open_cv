import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(1568, 128) # 32 * 7 * 7, 128
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


# creating a black white image of the given handwritten-number (picture)
kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (15,15))
img=cv2.imread('hand_writing_recognition/pictures/number2.jpg')
img_bgr=img.copy()
img_bgr=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray=img.copy()
img_gray=cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
retval, dst= cv2.threshold(img_gray, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(dst, cmap='gray')

# applying dilation and erosion to get the handwritten number blurred and resized to 28x28 type of MNIST, becasuse the model was trained on mnist pictures

dst=cv2.dilate(dst, kernel, iterations=15)
dst=cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
dst=cv2.erode(dst, kernel, iterations=5)
dst=cv2.resize(dst, (28,28))
filename='output.png'
cv2.imwrite(filename, dst)
plt.imshow(dst, cmap='gray')

dst_torch = torch.from_numpy(dst)


net = ConvNet()
net = torch.load('hand_writing_recognition/trained_model_15b.pth')
net.eval()

logits = net.forward(dst_torch.unsqueeze(0))
ps = F.softmax(logits, dim=1)
view_classify(dst_torch, ps)
print(f'The given Image was classified as: {torch.argmax(ps)}')

# contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# copy = img.copy()
# cv2.drawContours(copy, contours, -1,(0,255,0), 4)
# plt.imshow(copy)


# sorted_contours = sorted(contours, key = cv2.contourArea, reverse=True)
# copy = img.copy()
# cv2.drawContours(copy, sorted_contours[0:4], -1,(0,255,0), 4)
# plt.imshow(copy)


# xdd = []
# ydd = []
# wdd = []
# hdd = []
# padding = 4
# for i in sorted_contours[0:4]:
#     (xd, yd, wd, hd) = cv2.boundingRect(i)
#     # if (wd >= 0) and (hd >= 100):
#     xdd.append(xd)
#     ydd.append(yd)
#     wdd.append(wd)
#     hdd.append(hd)
# numbers = []
# for x, y, w, h in zip(xdd, ydd, wdd, hdd):
#     num = img[y:y + h, x:x + w]
#     numbers.append(num)
# k = 0
# for i in numbers:
#     print(i)
#     i = cv2.resize(i, (24, 24))
#     i = cv2.copyMakeBorder(i, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
#     ret, i = cv2.threshold(i, 143, 255, cv2.THRESH_BINARY_INV)
#     cv2.imwrite(f'img/{k}.jpg', i)
#     plt.imshow(i)
#     k += 1