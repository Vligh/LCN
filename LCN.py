# 作者：张鑫
# 时间：2022/8/26 15:31
#变红了，对比度极高
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os

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

    c, h, w =image.shape[0], image.shape[1], image.shape[2]  # (图片数、层数、x轴、y轴大小)

    gaussian_filter = torch.Tensor(get_gaussian_filter((1, c, radius, radius))).to(device)  # 创建卷积核
    filtered_out = F.conv2d(image, gaussian_filter, padding=radius - 1)  # 卷积 (∑ipq Wpq.X i,j+p,k+q)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))  # 获得卷积核的中心位置

    #g(x,y)=(f(x,y)-m_f(x,y)) / max(σ_f(x,y),c)
    #f(x,y)是输入图像，m_f(x,y)是局部均值，σ_f(x,y)是局部方差，c是局部方差均值，g(x,y)是输出图像

    ### Subtractive Normalization
    centered_image = image - filtered_out[ :, mid:-mid, mid:-mid]  # Vijk
    # ↑由于padding为radius-1,filered_out比实际图片要大,截取mid:-mid后才是有效区域)

    ## Variance Calc
    sum_sqr_image = F.conv2d(centered_image.pow(2), gaussian_filter, padding=radius - 1)  # ∑ipqWpq.v2 i,j+p,k+q
    s_deviation = sum_sqr_image[ :, mid:-mid, mid:-mid].sqrt()  # σ jk
    per_img_mean = s_deviation.mean()  # c

    ## Divisive Normalization
    divisor = np.maximum(per_img_mean.cpu().detach().numpy(), s_deviation.cpu().detach().numpy())  # max(c, σjk)
    divisor = np.maximum(divisor, 1e-4)
    new_image = centered_image / torch.Tensor(divisor).to(device)  # Yijk

    return new_image.to(device)

#####测试#####

# 显示tensor图片
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    #cv2.imwrite('LCN-Image/001.jpg', image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    ans=[]
    input_transform = transforms.Compose([transforms.ToTensor(),])

    path = '15_0508funnydunkey_crop1'
    files = os.listdir(path)
    index = 0
    for file in files:
        img = cv2.imread(path+"/{}".format(file), 0)  # 灰度化了
        img = torch.Tensor([np.array(img)]).to(device)
        img = local_contrast_norm(img, 3)
        img = img[0].cpu().numpy()
        img = img.astype(np.uint8)
        img = input_transform(img)
        out = np.transpose(img, (1, 2, 0))
        out = np.array(out)
        ans.append(out)
        #print("ans",index)
        #index = index + 1

    print(len(ans))#71
    print(sum(sum(ans[1] - ans[0])))#如果ans[1]与ans[0]的差异为最大值那么归一化后为1

    data = []
    for i in range(1,len(ans)):
        data.append(sum(sum(ans[i] - ans[0])))

    print(len(data))#70

    #线性归一化
    data_min = np.min(data)
    data_max = np.max(data)
    print(data_max,data_min)
    for i in range(0,len(data)):
        data[i] = (data[i]-data_min)/(data_max-data_min)
    plt.plot(data)
    plt.xlabel('frame')
    plt.ylabel('magnitude')
    # plt.legend()
    plt.show()#data[0]是1

    print(data[0],data[1],data[69])

    ''' 以下是单个图像的代码，并且最后每位相减，量化为单个数字
    img = cv2.imread("image/img001.jpg",0)#灰度化了
    img = torch.Tensor([np.array(img)]).to(device)
    img = local_contrast_norm(img, 3)
    img = img[0].cpu().numpy()
    img = img.astype(np.uint8)
    img = input_transform(img)
    out = np.transpose(img,(1,2,0))
    out = np.array(out)
    ans.append(out)
    cv2.imshow("1", out)
    cv2.waitKey(0)

    img2 = cv2.imread("image/img002.jpg",0)
    img2 = torch.Tensor([np.array(img2)]).to(device)
    img2 = local_contrast_norm(img2,3)
    img2 = img2[0].cpu().numpy()
    img2 = img2.astype(np.uint8)
    img2 = input_transform(img2)
    out2 = np.transpose(img2,(1,2,0))
    out2 = np.array(out2)
    ans.append(out2)
    cv2.imshow("2",out2)
    cv2.waitKey(0)

    print(ans[0].shape)
    print(sum(sum(ans[1]-ans[0])))
    #需要把这个LCN处理过的图像保存下来吗？直接使用，计算出每帧的LCN图，后续帧的LCN减去起始帧的LCN
    #应该和光流那种思想类似，将得到的差异值量化出来，看是否和微表情区间一致
    #cv2.imwrite("LCN-Image/img001.jpg")
    '''