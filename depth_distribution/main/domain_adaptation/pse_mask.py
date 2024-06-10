##PLW统计代码,运行self_train.py文件


import sys
import os
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import ctypes
import cv2
from ctypes import *
from PIL import Image
from depth_distribution.main.utils.misc import resize
from depth_distribution.main.utils import transformmasks 
from depth_distribution.main.utils import transformsgpu

# from depth_distribution.main.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from depth_distribution.main.utils.func import loss_calc, loss_calc_self, bce_loss, prob_2_entropy
from depth_distribution.main.utils.func import loss_berHu
from depth_distribution.main.utils.viz_segmask import colorize_mask
# from datetime import datetime
import datetime
from depth_distribution.main.domain_adaptation.eval_UDA import evaluate_domain_adaptation
from depth_distribution.main.utils.build import adjust_learning_rate
from torch.utils.tensorboard import SummaryWriter
from depth_distribution.main.utils.visualization import save_image
import numpy as np
writer = SummaryWriter(os.path.join("./runs/self_train",str(datetime.date.today())))
from operator import itemgetter
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
def strongTransform(device,parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(device), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)


    return data, target

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def print_losses(current_losses):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'  {full_string}')
class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=250, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss


def Write(current_losses,i_iter):
    for loss_name,loss_value in current_losses.items():
        writer.add_scalar(loss_name,loss_value,i_iter)

def selftrain_depdis(feature_extractor,classifier,aux, optimizer_fea,optimizer_cls,source_loader,target_loader, test_loader, cfg,depth_esti=False):
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE

    device = cfg.GPU_ID

    # interpolate output segmaps
    interp_source = nn.Upsample(
        size=(input_size_source[1],input_size_source[0]),
        mode="bilinear",
        align_corners=True,
    )

    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    best_miou = -1
    best_model = '-1'
    num_samples = len(test_loader)
    rms     = -1
    log_rms = -1
    abs_rel = -1
    sq_rel  = -1
    a1      = -1
    a2      = -1
    a3      = -1
    best_depth = float("-inf")
    MIN_DEPTH = 0
    MAX_DEPTH = 0

    # labels for adversarial training
    sourceloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)
    testloader_iter = enumerate(test_loader)

    #定义loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCELoss()
    max_iters = cfg.TRAIN.MAX_ITERS
    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=255).cuda()
    name = [0,0,0,0]
    from depth_distribution.main.utils.func import per_class_iu, fast_hist
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for i_iter in tqdm(range(len(target_loader))):
    # for i_iter in tqdm(range(len(test_loader))):
        # _,batch = sourceloader_iter.__next__()
        _, batch1 = targetloader_iter.__next__() #在训练集加载图像，
        # _, batch1 = testloader_iter.__next__() #在测试集加载图像
        # if depth_esti:
            # images_source, labels_source, depthvalue_source,_, _,prob_target= batch
            # depthvalue_source = depthvalue_source.cuda(device,non_blocking = True).long()
        # else:
            # images_source, labels_source, _, _,_ ,prob_target= batch
        # prob_target = prob_target[0]
        # print(prob_target)

        # images_source = images_source.cuda(device,non_blocking = True)
        # labels_source = labels_source.cuda(device,non_blocking = True).long()
        # images, _, _, _, label_pseudo = batch1

        images, labels, _, _, _ = batch1
        
        images= images.cuda(device,non_blocking=True)
        
        id_to_trainid = { 7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,23:9,24:10,25:11,26:12,28:13,32:14,33:15}
        label_copy = 255 * np.ones(labels.shape, dtype=np.float32)
        for k, v in id_to_trainid.items():
            label_copy[labels == k] = v
        label_copy = torch.from_numpy(label_copy).cuda(device)
        label_copy = label_copy.cpu()
        label_copy = label_copy.numpy()[0]
        # label_pseudo = label_pseudo.cuda(device,non_blocking=True).long()
        images_size = images.shape[-2:]
        label_size = labels.shape[-2:]
    
        images_fea = feature_extractor(images)
        images = images.cpu()
        if depth_esti:
            images_pred,pse_depth,_ = classifier(images_fea) 
            pse_depth = resize(
            input=pse_depth,
            size=images_size,
            mode = "bilinear",
            align_corners=False
        )
        else:
            images_pred,_,_ = classifier(images_fea) #伪标签
        # images_aux = aux(images_fea) 
        images_pred = resize(
            input=images_pred,
            size=label_size,
            mode='bilinear',
            align_corners=False)
        # label_pseudo = images_pred.max(1)[1]
        # if i_iter %cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN == 0 and int(best_model) ==  i_iter - 1:
        pred_trg_main_1 = F.softmax(images_pred, dim=1)
        conf, label_pseudo = torch.max(pred_trg_main_1, 1)
        conf = conf.cpu()
        mask = np.where(conf > 0.9 , 255 ,0)    #>0.9为白色（255）

        label_pseudo = label_pseudo.cpu()
        label_pseudo = label_pseudo.numpy()[0]
        # out_dir = '/home/ailab/ailab/SYN/Finally/Trans_depth3/depth_distribution/data/MLW'
        # mask1 = 255*np.ones(mask.shape,dtype=np.uint8) #<0.9,0 1,1024,2048
        # mask1 = np.repeat(mask1,3,axis=0) #3,1024,2048
        # mask1 = np.transpose(mask1,(1,2,0)) #1024,2048,3
        # print(mask1.shape)
        # print(mask1.shape)
        # exit()
        # mask2 = 255*np.ones(mask.shape,dtype=np.uint8) #>0.9,255
        # mask2 = np.repeat(mask2,3,axis=0) #3,1024,2048
        # mask2 = np.transpose(mask2,(1,2,0)) #1024,2048,3
        # print("***********mask",mask)
        # print("***************mask1",mask1.shape)
        #*****************统计代码*********************
        for i in range(label_size[0]):
            for j in range(label_size[1]):
                if(mask[0][i][j] == 255 and label_copy[i][j] == label_pseudo[i][j] and label_copy[i][j] !=255 ) : 
                    name[0]=name[0]+1 
                    # mask2[i][j] = [225,0,0] #>0.9,right
                elif(mask[0][i][j] == 255 and label_copy[i][j] != label_pseudo[i][j] and label_copy[i][j] !=255  ) :
                    name[1]=name[1]+1 
                    # mask2[i][j] = [225,0,225] #>0.9,error
                elif(mask[0][i][j] == 0 and label_copy[i][j] == label_pseudo[i][j] and label_copy[i][j] !=255  ) :
                    name[2] = name[2]+1 
                    # mask1[i][j] = [0,225,0] #<0.9,right,绿
                elif(mask[0][i][j] == 0 and label_copy[i][j] != label_pseudo[i][j] and label_copy[i][j] !=255  ) :
                    name[3] = name[3]+1 
                    # mask1[i][j] = [200,225,0] #<0.9,error，黄
        print("name[0]:",name[0],"name[1]:",name[1],"name[2]:",name[2],"name[3]:",name[3])
        print(">0.9",name[0]/(name[1]+name[0]))
        print("<0.9",name[2]/(name[3]+name[2]))
        exit()
        #*************统计代码结束*****************************
        # images = images.cpu()
        # import matplotlib.pyplot as plt
        # import cv2
        # from PIL import Image
        # #**********************保存图像、标签、阈值图可视化********************
        # save_image(out_dir, images[0], None, f'{i_iter}_0img')
        # save_image(out_dir, label_pseudo, None, f'{i_iter}_1pred')
        # save_image(out_dir, label_copy, None, f'{i_iter}_2gt')

    
        # # cv2.imwrite(os.path.join(out_dir,str(i_iter)+"_3small.png"),mask1)
        # im_1 = Image.fromarray(mask1,'RGB')
        # im_1.save(os.path.join(out_dir,str(i_iter)+"_3small.png"))
        # im_2 = Image.fromarray(mask2,'RGB')
        # im_2.save(os.path.join(out_dir,str(i_iter)+"_4big.png"))
        # exit()

        #**********************保存图像、标签、阈值图可视化结束********************
        #**********************标签和伪标签可视化**************
        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.title("labels")
        # plt.imshow(label_copy[0],cmap="gray") #,cmap="gray"
        # plt.subplot(1,2,2)
        # plt.title("label_pseudo")
        # plt.imshow(label_pseudo,cmap="gray") #,cmap="gray"
        # plt.show()
        #**********************标签和伪标签可视化结束**************
        # exit()
        #*******************计算IoU*****************
        # hist += fast_hist(label_copy.flatten(), label_pseudo.flatten(), cfg.NUM_CLASSES)
        # if i_iter > 0 and i_iter % 100 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(
        #         i_iter, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
        #*******************计算IoU结束******************
        # mask_out = np.array(mask[0],np.uint8)
        # import matplotlib.pyplot as plt
        # plt.title("mask")
        # plt.imshow(mask_out,cmap="gray")
        # plt.show()


        # print(mask)
        # result = np.where(mask ==255 ,50,150)
        # result_big = np.where(mask ==255 and )
        # mask_big = 50*np.ones(label_size) #>0.9的预测错了为50，预测对了为100
        # mask_small = 150*np.ones(label_size) #<0.9的预测错了为150，预测对了为150
        # for i in range(label_size[0]):
        #     for j in range(label_size[1]):
        #         print("go",i,j)
        #         if(labels[0][i][j] == label_pseudo[0][i][j] ):
        #             if mask[0][i][j] ==255:
        #                 result[0][i][j]=100
        #             else:
        #                 result[0][i][j]=200


        # #*****************统计代码*********************
        # for i in range(label_size[0]):
        #     for j in range(label_size[1]):
        #         if(mask[0][i][j] == 255 and label_copy[i][j] == label_pseudo[i][j] ) :name[0] = name[0]+1 #>0.9,right
        #         elif(mask[0][i][j] == 255 and label_copy[i][j] != label_pseudo[i][j] ) :name[1] = name[1]+1 #>0.9,error
        #         elif(mask[0][i][j] == 0 and label_copy[i][j] == label_pseudo[i][j] ) :name[2] = name[2]+1 #<0.9,right
        #         elif(mask[0][i][j] == 0 and label_copy[i][j] != label_pseudo[i][j] ) :name[3] = name[3]+1 #<0.9,error
        # print("name[0]:",name[0],"name[1]:",name[1],"name[2]:",name[2],"name[3]:",name[3])
        # #*************统计代码结束*****************************


        # result_out = np.array(result[0],np.uint8)
        # plt.imshow(result_out,cmap="gray")
        # plt.title("plw")
        # plt.show()
    # inters_over_union_classes = per_class_iu(hist)
    # computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    # print('\tCurrent mIoU:', computed_miou)


        

