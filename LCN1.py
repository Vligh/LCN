# 作者：张鑫
# 时间：2022/8/26 9:24
#图像感觉曝光了
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def local_contrast_norm(image, radius=9):
    """
    image: torch.Tensor , .shape => (1,channels,height,width)
    radius: Gaussian filter size (int), odd
    """
    if radius % 2 == 0:  # LCN核的大小为奇数
        radius += 1
    def get_gaussian_filter(kernel_shape):
        x = np.zeros(kernel_shape, dtype='float64')
        # 二维高斯函数
        def gauss(x, y, sigma=2.0):
            Z = 2 * np.pi * sigma ** 2
            return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
        mid = np.floor(kernel_shape[-1] / 2.)  # 求出卷积核的中心位置(mid,mid)
        for kernel_idx in range(0, kernel_shape[1]):  # 遍历每一层
            for i in range(0, kernel_shape[2]):  # 遍历x轴
                for j in range(0, kernel_shape[3]):  # 遍历y轴
                    x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)  # 计算出高斯权重
        return x / np.sum(x)
    n, c, h, w = 1, image.shape[0], image.shape[1], image.shape[2]  # (图片数、层数、x轴、y轴大小)
    gaussian_filter = torch.Tensor(get_gaussian_filter((1, c, radius, radius))).to(device)  # 创建卷积核
    filtered_out = F.conv2d(image, gaussian_filter, padding=radius - 1) # 卷积 (∑ipq Wpq.X i,j+p,k+q)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))  # 获得卷积核的中心位置

    ### Subtractive Normalization
    centered_image = image - filtered_out[:, mid:-mid, mid:-mid]  # Vijk
    # ↑由于padding为radius-1,filered_out比实际图片要大,截取mid:-mid后才是有效区域)

    ## Variance Calc
    sum_sqr_image = F.conv2d(centered_image.pow(2), gaussian_filter, padding=radius - 1)  # ∑ipqWpq.v2 i,j+p,k+q
    s_deviation = sum_sqr_image[:, mid:-mid, mid:-mid].sqrt()  # σ jk
    per_img_mean = s_deviation.mean()  # c

    ## Divisive Normalization
    divisor = np.maximum(per_img_mean.cpu().detach().numpy(), s_deviation.cpu().detach().numpy())  # max(c, σjk)
    divisor = np.maximum(divisor, 1e-4)
    new_image = centered_image / torch.Tensor(divisor).to(device)  # Yijk

    return new_image.to(device)

img = cv2.imread("image/img005.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#print(img)
img_chw = np.transpose(img,(2,0,1))
trans = transforms.ToTensor()
img_tensor = trans(img_chw)
img_cuda = img_tensor.cuda()

out = local_contrast_norm(img_cuda)
out = out.detach().cpu().numpy()
out = np.transpose(out,(2,0,1))

print(out)
cv2.imshow("lcn_out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
