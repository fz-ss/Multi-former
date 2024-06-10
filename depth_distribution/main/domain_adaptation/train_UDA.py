import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import os
from depth_distribution.main.utils.misc import resize
# from depth_distribution.main.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from depth_distribution.main.utils.func import loss_calc, prob_2_entropy
from depth_distribution.main.utils.func import loss_berHu
from depth_distribution.main.utils.denpro import getTargetDensity_16, getTargetDensity_7, getTargetDensity_7_small, getSourceDensity
from depth_distribution.main.domain_adaptation.eval_UDA import evaluate_domain_adaptation
from depth_distribution.main.utils.build import adjust_learning_rate
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from operator import itemgetter
import datetime
writer = SummaryWriter(os.path.join("./runs/train",str(datetime.date.today())))

def get_share_weight(domain_out, before_softmax,  class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = F.softmax(before_softmax, dim=1)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    return weight.detach()

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*torch.log(pred)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'  {full_string}')

def Write(current_losses,i_iter):
    for loss_name,loss_value in current_losses.items():
        writer.add_scalar(loss_name,loss_value,i_iter)

def train_depdis(feature_extractor,classifier,aux, model_D,model_Dis, optimizer_fea,optimizer_cls, optimizer_D,  source_loader, target_loader,test_loader, cfg, expid, iternum, distributed, local_rank):
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    batch_size = cfg.TRAIN.BATCH_SIZE_SOURCE
    if distributed:
        device = torch.device(cfg.DEVICE)
    else:
        device = cfg.GPU_ID
    if distributed:
        batch_size = int(cfg.TRAIN.BATCH_SIZE_SOURCE / torch.distributed.get_world_size())
        print("torch.distributed.get_world_size();torch.distributed.get_world_size()",torch.distributed.get_world_size())
        classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        aux = torch.nn.SyncBatchNorm.convert_sync_batchnorm(aux)
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        aux = torch.nn.parallel.DistributedDataParallel(
            aux, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg3
        )
        pg4 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg4
        )
        pg5 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_Dis = torch.nn.parallel.DistributedDataParallel(
            model_Dis, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg5
        ) 
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    cudnn.benchmark = True
    cudnn.enabled = True

    # interpolate output segmaps
    interp_source = nn.Upsample(
        size=(input_size_source[1], input_size_source[0]),
        mode="bilinear",
        align_corners=True,
    )
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    source_label = 0
    target_label = 1

    best_miou = -1
    best_model = '-1'
    MAX_DEPTH = 50 #50a
    MIN_DEPTH = 1e-3

    trainloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)

    #定义loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCELoss()
    max_iters = cfg.TRAIN.MAX_ITERS
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS)) :

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, i_iter, 1500, 1e-6, max_iters)
        current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, i_iter, 0, 0, max_iters, power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_D.param_groups)):
            optimizer_D.param_groups[index]['lr'] = current_lr_D


        n_iter, batch = trainloader_iter.__next__()
        images_source, labels_source, depthvalue_source,_, _,_= batch
        src_size = images_source.shape[-2:]

        _, batch1 = targetloader_iter.__next__()
        images_target, _, _, _, _= batch1
        tgt_size = images_target.shape[-2:]

        if n_iter <= iternum and iternum != 0:
            continue

        # UDA Training

        #feature_extractor and classifier network
        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        images_source = images_source.cuda(device,non_blocking=True)
        labels_source = labels_source.cuda(device,non_blocking=True).long()
        images_target = images_target.cuda(device,non_blocking=True)
        src_fea = feature_extractor(images_source)
        src_pred , pred_depthvalue_src ,pred_depthmix_src = classifier(src_fea)
        pred_depthvalue_src = interp_source(pred_depthvalue_src)
        loss_depth_src = loss_berHu(pred_depthvalue_src, depthvalue_source, device)

        src_aux = aux(src_fea) 
        src_pred = resize(
            input=src_pred,
            size=src_size,
            mode='bilinear',
            align_corners=False)
        src_aux = resize(
            input=src_aux,
            size=src_size,
            mode='bilinear',
            align_corners=False)
        loss_seg_src = 1.0*criterion(src_pred,labels_source)
        loss_aux_src = 0.4*criterion(src_aux,labels_source)

        src_fea_D = src_fea[-2]
        # src_fea_D = (src_fea_D + pred_depthmix_src ) / 2
        src_fea_D = prob_2_entropy(F.softmax(src_fea_D)) * pred_depthmix_src*pred_depthmix_src

        # cout1 = 
        src_Dis_pred = model_Dis(src_fea_D.detach(), src_size)
        source_share_weight = get_share_weight(src_Dis_pred, src_pred, class_temperature=10.0)
        source_share_weight = normalize_weight(source_share_weight)
        src_D_pred = model_D(src_fea_D, src_size)
        if cfg.SOLVER.DIS == 'binary':
            loss_adv_src = 0.001*soft_label_cross_entropy((torch.ones_like(src_D_pred)-src_D_pred).clamp(min=1e-7, max=1.0), torch.ones_like(src_D_pred),source_share_weight)
        # else:
        #     loss_adv_src = 0.001*soft_label_cross_entropy(F.softmax(src_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((src_soft_label,torch.zeros_like(src_soft_label)), dim=1),source_share_weight)

        losses = cfg.TRAIN.LAMBDA_SEG_SRC * loss_seg_src + cfg.TRAIN.LAMBDA_SEG_SRC * loss_aux_src +cfg.TRAIN.LAMBDA_ADV_TAR * loss_adv_src+ cfg.TRAIN.LAMBDA_DEP_SRC * loss_depth_src
        losses.backward()
        # (cfg.TRAIN.LAMBDA_SEG_SRC * loss_seg_src + cfg.TRAIN.LAMBDA_SEG_SRC * loss_aux_src +cfg.TRAIN.LAMBDA_ADV_TAR * loss_adv_src+ cfg.TRAIN.LAMBDA_DEP_SRC * loss_depth_src).backward()
        optimizer_fea.step()
        optimizer_cls.step()

        #discriminator network
        tgt_fea = feature_extractor(images_target)
        tgt_pred,_,pred_depthmix_tgt = classifier(tgt_fea)
        tgt_pred = resize(
            input=tgt_pred,
            size=tgt_size,
            mode='bilinear',
            align_corners=False)
        optimizer_D.zero_grad()

        src_fea_D = src_fea[-2]
        # src_fea_D = (src_fea_D + pred_depthmix_src ) / 2
        src_fea_D = prob_2_entropy(F.softmax(src_fea_D))*pred_depthmix_src*pred_depthmix_src
        src_Dis_pred = model_Dis(src_fea_D.detach(), src_size)
        loss_Dis_src = 0.5*bce_loss(src_Dis_pred, torch.ones_like(src_Dis_pred))
        loss_Dis_src.backward()

        tgt_fea_D = tgt_fea[-2]
        # tgt_fea_D = (tgt_fea_D + pred_depthmix_tgt ) / 2
        tgt_fea_D = prob_2_entropy(F.softmax(tgt_fea_D)) * pred_depthmix_tgt*pred_depthmix_tgt
        tgt_Dis_pred = model_Dis(tgt_fea_D.detach(), tgt_size)
        loss_Dis_tgt = 0.5*bce_loss(tgt_Dis_pred, torch.zeros_like(tgt_Dis_pred))
        loss_Dis_tgt.backward()

        source_share_weight = get_share_weight(src_Dis_pred, src_pred, class_temperature=10.0)
        source_share_weight = normalize_weight(source_share_weight)
        target_share_weight = -get_share_weight(tgt_Dis_pred, tgt_pred, class_temperature=1.0)
        target_share_weight = normalize_weight(target_share_weight)

        src_D_pred = model_D(src_fea_D.detach(), src_size)
        if cfg.SOLVER.DIS == 'binary':
            loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred.clamp(min=1e-7, max=1.0), torch.ones_like(src_D_pred), source_share_weight)
        else:
            loss_D_src = 0.5*soft_label_cross_entropy(F.softmax(src_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((torch.zeros_like(src_soft_label),src_soft_label), dim=1), source_share_weight)
        loss_D_src.backward()

        tgt_D_pred = model_D(tgt_fea_D.detach(), tgt_size)
        if cfg.SOLVER.DIS == 'binary':
            loss_D_tgt = 0.5*soft_label_cross_entropy((torch.ones_like(tgt_D_pred)-tgt_D_pred).clamp(min=1e-7, max=1.0), torch.ones_like(tgt_D_pred), target_share_weight)
        else:
            loss_D_tgt = 0.5*soft_label_cross_entropy(F.softmax(tgt_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((tgt_soft_label,torch.zeros_like(tgt_soft_label)), dim=1), target_share_weight)
        loss_D_tgt.backward()

        optimizer_D.step()

        current_losses = {
            'loss_seg_src': loss_seg_src,
            'loss_aux_src': loss_aux_src,
            'loss_adv_src':loss_adv_src,
            "loss_depth_src":loss_depth_src,
            'loss_Dis_trg': loss_Dis_src,
            'loss_Dis_tgt': loss_Dis_tgt,
            'loss_D_src':loss_D_src,
            'loss_D_tgt':loss_D_tgt
        }
        Write(current_losses,i_iter)

        # print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:

            print("taking snapshot ...")
            print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f"{i_iter}.pth")
            if distributed==False or (distributed==True and dist.get_rank()) == 0:
                torch.save({'i_iter': i_iter, 
                'feature_extractor': feature_extractor.state_dict(), 
                'classifier':classifier.state_dict(), 
                'aux':aux.state_dict(),
                'model_D': model_D.state_dict(), 
                'model_Dis': model_Dis.state_dict()
                }, snapshot_dir)

                # eval model, can be annotated
                feature_extractor.eval()
                classifier.eval()
                model_D.eval()
                model_Dis.eval()
                lastbest_model = best_model
                best_miou, best_model ,current_miou,_,_,_,_,_,_,_,_= evaluate_domain_adaptation(feature_extractor,classifier, test_loader, cfg, i_iter, best_miou, best_model,device,False,1,1,1,1,1,1,1,1,MIN_DEPTH,MAX_DEPTH)
                writer.add_scalar("miou",current_miou,i_iter)
                print(f"best_model{best_model},last_model{i_iter-cfg.TRAIN.SAVE_PRED_EVERY}")
                print(type(best_model))
                print(type(i_iter-cfg.TRAIN.SAVE_PRED_EVERY))
                print(best_model != i_iter-cfg.TRAIN.SAVE_PRED_EVERY)
                if int(best_model) != i_iter-cfg.TRAIN.SAVE_PRED_EVERY and i_iter != cfg.TRAIN.SAVE_PRED_EVERY:  #remove last_model
                    print(f"{i_iter}first，remove{i_iter-cfg.TRAIN.SAVE_PRED_EVERY}.pth")
                    if os.path.exists(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f"{i_iter-cfg.TRAIN.SAVE_PRED_EVERY}.pth")):
                        os.remove(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f"{i_iter-cfg.TRAIN.SAVE_PRED_EVERY}.pth"))
                print(f"best_model:{best_model},i_iter:{i_iter},best_miou:{best_miou}")
                print(best_model == i_iter)
                if int(lastbest_model)!= -1 and i_iter != cfg.TRAIN.SAVE_PRED_EVERY and int(best_model) == i_iter and int(lastbest_model) != i_iter-cfg.TRAIN.SAVE_PRED_EVERY :  #remove lastbest_model
                    if os.path.exists(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f"{lastbest_model}.pth")):
                        os.remove(os.path.join(cfg.TRAIN.SNAPSHOT_DIR,f"{lastbest_model}.pth"))                
                    print(f"{i_iter}second,remove{lastbest_model}")
                feature_extractor.train()
                classifier.train()
                model_D.train()
                model_Dis.train()


            if i_iter >= cfg.TRAIN.EARLY_STOP:
                break

        sys.stdout.flush()
    print(f' best mIoU:{best_miou}\n')
    print(f' best model:{best_model}\n','.pth')