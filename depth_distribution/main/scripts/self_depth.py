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
from ctypes import *
from PIL import Image
from depth_distribution.main.utils.misc import resize
from depth_distribution.main.utils import transformmasks 
from depth_distribution.main.utils import transformsgpu
import os
from depth_distribution.main.utils.visualization import save_image
# from depth_distribution.main.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from depth_distribution.main.utils.func import loss_calc, loss_calc_self, bce_loss, prob_2_entropy
from depth_distribution.main.utils.func import loss_berHu
from depth_distribution.main.utils.viz_segmask import colorize_mask
# from datetime import datetime
import datetime
from depth_distribution.main.domain_adaptation.eval_UDA import evaluate_domain_adaptation
from depth_distribution.main.utils.build import adjust_learning_rate
from torch.utils.tensorboard import SummaryWriter
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
    MAX_DEPTH =50 #30#20 #16  #50 #50a
    MIN_DEPTH = 1e-3

    # labels for adversarial training
    sourceloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)

    #定义loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCELoss() 
    max_iters = cfg.TRAIN.MAX_ITERS
    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=255).cuda()

    for i_iter in tqdm(range(cfg.MAX_ITERS_SELFTRAIN)):
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, i_iter, 1500, 1e-6, max_iters)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr
        try:
            _,batch = sourceloader_iter.__next__()
            _, batch1 = targetloader_iter.__next__()
        except StopIteration:
            sourceloader_iter = enumerate(source_loader)
            targetloader_iter = enumerate(target_loader)
            _,batch = sourceloader_iter.__next__()
            _, batch1 = targetloader_iter.__next__()
        if depth_esti:
            images_source, _, depthvalue_source,_, _,prob_target= batch
            depthvalue_source = depthvalue_source.cuda(device,non_blocking = True).long()
        else:
            images_source, labels_source, _, _,_ ,prob_target= batch
            labels_source = labels_source.cuda(device,non_blocking = True).long()
        prob_target = prob_target[0]
        # print(prob_target)

        images_source = images_source.cuda(device,non_blocking = True)
       
        # images, _, _, _, label_pseudo = batch1
        if depth_esti:
            target_iter = iter(target_loader)
            images, _, depth_trg, _, _ = next(target_iter)
            interp = nn.Upsample(size=(depth_trg.shape[1], depth_trg.shape[2]), mode='bilinear', align_corners=True)

        else:
            images, _, _, _, _ = batch1
        
        images= images.cuda(device,non_blocking=True)
        # label_pseudo = label_pseudo.cuda(device,non_blocking=True).long()
        images_size = images.shape[-2:]
    
        images_fea = feature_extractor(images)
        if depth_esti:
            _,pse_depth,_ = classifier(images_fea)
            # pse_depth = resize(
            # input=pse_depth,
            # size=(depth_trg.shape[1],depth_trg.shape[2]),
            # mode = "bilinear",
            # align_corners=False
            # )

            pse_depth = interp(pse_depth)
            depth_trg = depth_trg.squeeze()
            pse_depth = pse_depth.squeeze()
            pse_depth =pse_depth.cpu()
            mask = np.logical_and(depth_trg > MIN_DEPTH, depth_trg  < MAX_DEPTH)
            w = depth_trg .shape[1]
            h = depth_trg .shape[0]
            crop = np.array([0.40810811 * h, 0.99189189 * h,
                            0.03594771 * w,  0.96405229 * w]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            depth_trg = depth_trg[mask]
            pse_depth = pse_depth[mask]
            pse_depth.cuda(device)
            depth_trg.cuda(device)

        else:
            images_pred,_,_ = classifier(images_fea)
        # images_aux = aux(images_fea) 
            images_pred = resize(
                input=images_pred,
                size=images_size,
                mode='bilinear',
                align_corners=False)
            # label_pseudo = images_pred.max(1)[1]
            # if i_iter %cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN == 0 and int(best_model) ==  i_iter - 1:
            pred_trg_main_1 = F.softmax(images_pred, dim=1)
            conf, label_pseudo = torch.max(pred_trg_main_1, 1)
            mask_tgt = np.where(conf > 0.9 , 0.9 ,0.5)       
            # print(mask)
            mask_src = np.ones(mask_tgt.shape)
            mask_tgt = torch.tensor(mask_tgt).cuda(device)
            mask_src = torch.tensor(mask_src).cuda(device)
        

        #feature_extractor and classifier network
        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        if depth_esti ==False:

            for image_i in range(cfg.TRAIN.BATCH_SIZE_TARGET):

                # labels[img_i].shape = [512, 512], only get one pic,here souce images
                classes = torch.unique(labels_source[image_i])  # selet unique data classes = [16]
                nclasses = classes.shape[0]  # 16
                ########################DO NOT USE RCM
                
               #  classes = (classes[torch.Tensor(np.random.choice(nclasses ,int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda(device) #random choose half classes
               ######
               #  classes = torch.tensor([4, 10, 8, 2, 13, 5, 1])
               #  classes=classes[classes!=ignore_label] 
                #######################USE RCM
                classes = sorted(classes)
                if(classes[-1]==255) : del classes[-1]
                classes = torch.tensor(classes)
                prob_target_iter = []
                for i in classes:
                   prob_target_iter.append( itemgetter(i)(prob_target) )
                prob_target_iter = np.array(prob_target_iter)
                prob_target_iter = prob_target_iter/prob_target_iter.sum()
                classes_choose = []
                while(True):
                    temp =np.random.choice(classes, p=prob_target_iter)
                    if(classes_choose.__contains__(temp)):
                       continue
                    classes_choose.append(temp)
                    if(len(classes_choose) == int((nclasses+nclasses%2)/2)):break
                classes_choose = torch.tensor(classes_choose).long().cuda(device)
                classes_choose.cuda(device)
               
                # MixMask ust mix some classes and output the one predit
               #  if image_i == 0:  # labels[image_i] = tensor([512, 512])
               #      MixMask0 = transformmasks.generate_class_mask(labels_source[image_i], classes).unsqueeze(0).cuda(device)
               #  else:
               #      MixMask1 = transformmasks.generate_class_mask(labels_source[image_i], classes).unsqueeze(0).cuda(device)
                if image_i == 0:  # labels[image_i] = tensor([512, 512])
                    MixMask0 = transformmasks.generate_class_mask(labels_source[image_i], classes_choose).unsqueeze(0).cuda(device)
                else:
                    MixMask1 = transformmasks.generate_class_mask(labels_source[image_i], classes_choose).unsqueeze(0).cuda(device)
                #
                # MixMask1 shape = [1, 512, 512], only all mask , 0 or 1`
                       # mix source img[0] and target img_reamin[0], then do aug on the result
            strong_parameters = {"Mix": MixMask0}
            strong_parameters["flip"] = 0
            strong_parameters["ColorJitter"] = random.uniform(0, 1)
            strong_parameters["GaussianBlur"] = random.uniform(0, 1)

            #mix picture
            strong_parameters["Mix"] =MixMask0
            image_mix1 ,_ = strongTransform(device,strong_parameters,data = torch.cat((images_source[0].unsqueeze(0),images[0].unsqueeze(0))))
            mix_fea = feature_extractor(image_mix1)
            mix_pred,_,_ = classifier(mix_fea)
            mix_aux = aux(mix_fea)
            mix_pred  = resize(
                input = mix_pred,
                size = images_size,
                mode = 'bilinear',
                align_corners=False)
            mix_aux = resize(
                input = mix_aux,
                size = images_size,
                mode='bilinear',
                align_corners=False)

            label_mix ,_ = strongTransform(device,strong_parameters,data = torch.cat((labels_source[0].unsqueeze(0),label_pseudo[0].unsqueeze(0))))
            mask_mix,_ = strongTransform(device,strong_parameters,data = torch.cat((mask_src[0].unsqueeze(0),mask_tgt[0].unsqueeze(0))))
            loss_seg_trg =  1.0*unlabeled_loss(mix_pred,label_mix,mask_mix)
            loss_aux_trg = 0.4*unlabeled_loss(mix_aux,label_mix,mask_mix)
            loss_seg_trg.requires_grad_(True)
            loss_aux_trg.requires_grad_(True)
        if depth_esti:
            # loss_depth_src = loss_berHu(pre_depth,depthvalue_source,device)
            loss_depth_trg = loss_berHu(pse_depth, depth_trg, device)
        # (loss_seg_trg+loss_aux_trg+loss_seg_src+loss_aux_src).backward()
        else:
            (loss_seg_trg+loss_aux_trg).backward()
        if depth_esti:
             loss_depth_trg.backward()
        optimizer_fea.step()


        optimizer_cls.step()
        if depth_esti:
            current_losses = {
                "loss_depth_trg":loss_depth_trg
            }
        else:

            current_losses = {
                "loss_seg_trg": loss_seg_trg,
                "loss_aux_trg":loss_aux_trg
            }
        # print_losses(current_losses)
        Write(current_losses,i_iter)

        if (i_iter+1) % cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN == 0 and i_iter != 0:
            print("MAX_DEPTH,MAX_DEPTH,MAX_DEPTH",MAX_DEPTH)
            print("taking snapshot ...")
            print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f"{i_iter}.pth")
            torch.save({'i_iter': i_iter, 
            'feature_extractor': feature_extractor.state_dict(), 
            'classifier':classifier.state_dict(), 
            'aux':aux.state_dict(),
            }, snapshot_dir)
             # eval model, can be annotated
            feature_extractor.eval()
            classifier.eval()
            lastbest_model = best_model
            best_miou, best_model,current_miou,abs_rel,sq_rel,rms,log_rms,a1,a2,a3,best_depth = evaluate_domain_adaptation(feature_extractor,classifier, test_loader, cfg, i_iter, best_miou, best_model,device,depth_esti,rms,log_rms,abs_rel,sq_rel,a1,a2,a3,best_depth,MIN_DEPTH,MAX_DEPTH)
            writer.add_scalar("miou",current_miou,i_iter)
            print(f"i_iter:{i_iter}")
            print(f"best_model:{best_model},i_iter-cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN:{i_iter-cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN},best_miou:{best_miou}")
            print(int(best_model) != i_iter-cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN)
            if depth_esti:
               print("best depth")
               print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
               print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))

            if (i_iter+1)!= cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN and int(best_model)!= i_iter-cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN :  #remove last_model
                print(i_iter-cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN,"first remove")
                os.remove(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f"{i_iter-cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN}.pth"))
            if int(lastbest_model) != -1 and int(best_model) == i_iter and int(lastbest_model)!=i_iter-cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN and (i_iter+1) != cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN:  #remove lastbest_model
                print(i_iter,"second remove")
                os.remove(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f"{lastbest_model}.pth"))
            feature_extractor.train()
            classifier.train()

        sys.stdout.flush()
    print(f' best mIoU:{best_miou}\n')
    print(f' best model:{best_model}\n')
    if depth_esti:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
        print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))
