# 作者：张鑫
# 时间：2022/8/31 9:54
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

radius = 3
n_points = 8*radius

image = cv2.imread("image/img001.jpg")
#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",img_gray)
cv2.waitKey(0)

lbp = local_binary_pattern(img_gray,n_points,radius)
cv2.imshow("lbp",lbp)
cv2.waitKey(0)