import numpy as np
import torch
# # arr = [True,False,True]
# # print(np.sum(arr))
# # arr1 = np.ones(789665423,dtype=bool)
# # print(arr1)
# # print(np.sum(arr1))


# arr = np.random.rand(200)

# # print(arr.shape)
# # # a1 = arr <0.4
# # # print(a1)
# # # print(np.sum(a1))
# ground_truth = np.random.rand(32,64)
# predication=np.random.rand(32,64)

# rmse_log = (np.log(ground_truth) - np.log(predication)) ** 2
# rmse_log = np.sqrt(rmse_log.mean())
# print(rmse_log)
# a = np.zeros(7)
# a[3]=4
# print(a)
# classes_choose = []
# while(True):
#     temp =np.random.choice([1,2,3],p = [0.3,0.3,0.4])
#     print(temp)
#     if(classes_choose.__contains__(temp)):
#         continue
#     classes_choose.append(temp)
#     if(len(classes_choose)==3):break
# # print(classes_choose)
# a = torch.tensor([4, 10, 8, 2, 13, 5, 1])
# a = sorted(a)
# print(a)
# a = torch.tensor(a)
# print(a.shape)
# print(a.shape)
# a = a[0]
# print(a.shape)

 
# a=[1,2,3,4,5,67,8,9]
# c = [3,6]
# from operator import itemgetter
# b = []
# for i in c:

#     b.append( itemgetter(i)(a) )
# print(b)
# if(a[-1]==9) :a[-1]=None
# print(a)

# a = np.random.rand(1,640,320)
# print(a)
# mask = np.where(a>0.9 , 0.9 ,0.5)
# print(mask)
# mask_src = np.ones(mask.shape)
# print(mask_src)
# a = torch.tensor(a)
# b = len(a[1])
# print(b)
# import torch
# import torch.nn as nn
# inputs = torch.FloatTensor([0,1,2,0,0,0,0,0,0,1,0,0.5])
# outputs = torch.LongTensor([0,1,2,2])
# inputs = inputs.view((1,3,4))
# outputs = outputs.view((1,4))
# weight_CE = torch.FloatTensor([1,2,3])
# ce = nn.CrossEntropyLoss(weight=weight_CE)
# # ce = nn.CrossEntropyLoss(ignore_index=255)
# loss = ce(inputs,outputs)
# print(loss)

# a = True
# if ~a:
#     print("success")














# import torch
# from torch import nn
# import numpy as np
# input = torch.randn(1,512,10,20)
# model = nn.Sequential(
#     nn.AdaptiveAvgPool2d(2),
#     nn.Conv2d(512,2,2),
#     nn.BatchNorm2d(2),
#     nn.ReLU()
# )
# # model =nn.BatchNorm2d(2)
# output = model(input)
# print(output)

# from mmcv.cnn import ConvModule
# model = ConvModule(
#     512,
#     512,
#     1,
#     conv_cfg = None,
#     norm_cfg = dict(type='BN', requires_grad=True),
#     act_cfg = None
# )
# output= model(input)
# print(output)

from PIL import Image
import numpy as np

# 假设你有一个颜色表（palette），这里使用一些示例颜色
# palette = [
#     255, 0, 0,   # 红色
#     0, 255, 0,   # 绿色
#     0, 0, 255,   # 蓝色
#     # ... 其他颜色
# ]

# # 假设你有一个图像的颜色索引数据（mask_data），这里使用一个示例的二维数组
# # 注意：这里的 mask_data 是一个包含颜色索引的数组，不是 RGB 图像
# mask_data = np.array([[0, 1, 2], [1, 2, 0]])

# # 创建一个 Image 对象，并将颜色表应用于它
# mask_mat = Image.fromarray(mask_data).convert('P')
# mask_mat.putpalette(palette)

# # 将 Image 对象转换为数组
# mask_array = np.array(mask_mat)

# # 打印转换后的数组
# print("转换后的数组：")
# print(mask_array)
from PIL import Image
import numpy as np

# 假设你已经有一个颜色表（palette）和一个图像对象（image）
# 这里使用示例颜色表和图像
# palette = [
#     255, 0, 0,   # 红色
#     0, 255, 0,   # 绿色
#     0, 0, 255,   # 蓝色
#     # ... 其他颜色
# ]
palette = [
    3,
    7,
    9,   # 红色
    23,
    52,
    25,   # 绿色
    5,
    6,
    46  # 蓝色
    # ... 其他颜色
]

# 假设你的图像对象是一个 PIL Image 对象
# 如果不是 RGB 模式，先将其转换为 RGB 模式
image = np.array([[0, 1, 2], [1, 2, 0]])
# print(image.shape)
# exit()
image = Image.fromarray(image).convert('P')
# image = image.convert("RGB")
image.putpalette(palette)
image = image.convert("RGB")
# image.show()
# 获取图像的像素值
image_pixels = np.array(image)

# 打印图像的像素值
print("图像的像素值：")
print(image_pixels)
