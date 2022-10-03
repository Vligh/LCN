# 作者：张鑫
# 时间：2022/8/29 19:23
import cv2
import numpy as np

img = cv2.imread("image/img001.jpg")
#(3,3)表示高斯滤波器的长和宽都为3，1.3表示滤波器的标准差
out = cv2.GaussianBlur(img,(3,3),0)
cv2.imshow("out",out)
cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite("GaaussImage/Gauss001.jpg",out)