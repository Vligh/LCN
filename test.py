# 作者：张鑫
# 时间：2022/8/30 17:01
import cv2

if __name__ == '__main__':
    img = cv2.imread("image/img001.jpg",0)
    cv2.imshow("1",img)
    cv2.waitKey(0)