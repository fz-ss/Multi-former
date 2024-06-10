import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from depth_distribution.main.utils.func import per_class_iu, fast_hist
from depth_distribution.main.utils.misc import resize
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from depth_distribution.main.utils.dataset_util import compute_errors
# from datetime import datetime
import datetime
writer = SummaryWriter(os.path.join("./runs/eval",str(datetime.date.today())))

def evaluate_domain_adaptation( feature_extractor,classifier, test_loader, cfg, i_iter, best_miou, best_model,device,depth_esti,rms,log_rms,abs_rel,sq_rel,a1,a2,a3,best_depth,MIN_DEPTH,MAX_DEPTH):
    # device = cfg.GPU_ID
    # eval
    if cfg.TEST.MODE == 'best':
        best_miou, best_model ,iter_miou,best_abs_rel,best_sq_rel,best_rms,best_log_rms,best_a1,best_a2,best_a3 ,best_depth= eval_best(cfg, feature_extractor,classifier, device, test_loader, i_iter, best_miou, best_model,depth_esti,rms,log_rms,abs_rel,sq_rel,a1,a2,a3,best_depth,MIN_DEPTH,MAX_DEPTH)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")
    return best_miou, best_model,iter_miou,best_abs_rel,best_sq_rel,best_rms,best_log_rms,best_a1,best_a2,best_a3,best_depth

def record_result(text):
    print(text)
    with open("record.txt", "a") as f:
        f.write(text)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict['model'])
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        record_result(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)) + '\n')


def eval_best(cfg, feature_extractor,classifier,device, test_loader, i_iter, best_miou, best_model,depth_esti,rms,log_rms,abs_rel,sq_rel,a1,a2,a3,best_depth,MIN_DEPTH,MAX_DEPTH):
    cur_best_miou = best_miou
    cur_best_model = best_model
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = iter(test_loader)
   
    num_samples = len(test_loader)
    rms2     = np.zeros(num_samples, np.float32)
    log_rms2 = np.zeros(num_samples, np.float32)
    abs_rel2 = np.zeros(num_samples, np.float32)
    sq_rel2  = np.zeros(num_samples, np.float32)
    a12      = np.zeros(num_samples, np.float32)
    a22      = np.zeros(num_samples, np.float32)
    a32      = np.zeros(num_samples, np.float32)
        # MAX_DEPTH = 80 #50a
        # MIN_DEPTH = 1e-3
    for index in tqdm(range(len(test_loader))): #
        image, label, gt_depth,_,_ = next(test_iter)
        # image, label, _, name = next(test_iter)
        interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            pred ,depth,_= classifier(feature_extractor(image.cuda(device)))  #1,7,80,160

            pred = interp(pred)
            # pred = resize(   #1,7,1024,2048
            #     input=pred,
            #     size=label.shape[-2:],
            #     mode='bilinear',
            #     align_corners=False)
            output = pred.max(1)[1]  #1,1024,2048
            output=output.cpu()
            output = output.numpy()[0]
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        if index > 0 and index % 100 == 0:
            record_result('{:d} / {:d}: {:0.2f}\n'.format(index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            # break
        if depth_esti:
                interp = nn.Upsample(size=(gt_depth.shape[1], gt_depth.shape[2]), mode='bilinear', align_corners=True)
                pred_depth = interp(depth)
                gt_depth = gt_depth.squeeze()
                pred_depth = pred_depth.squeeze()
                pred_depth =pred_depth.cpu()
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                w = gt_depth.shape[1]
                h = gt_depth.shape[0]
                crop = np.array([0.40810811 * h, 0.99189189 * h,
                                0.03594771 * w,  0.96405229 * w]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
                gt_depth = gt_depth[mask]
                pred_depth = pred_depth[mask]

                abs_rel2[index], sq_rel2[index], rms2[index], log_rms2[index], a12[index], a22[index], a32[index] = compute_errors(gt_depth, pred_depth)
    inters_over_union_classes = per_class_iu(hist)
    # for i in range(cfg.NUM_CLASSES):
    #     writer.add_scalar("iou"+str(i),inters_over_union_classes[i],i_iter)
    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    if cur_best_miou < computed_miou:
        cur_best_miou = computed_miou
        cur_best_model =  f'{i_iter}'
    if depth_esti:
        cur_abs_rel = abs_rel2.mean()
        cur_sq_rel = sq_rel2.mean()
        cur_rms = rms2.mean()
        cur_log_rms = log_rms2.mean()
        cur_a1 = a12.mean()
        cur_a2 = a22.mean()
        cur_a3 = a32.mean()
        computed_depth = -cur_abs_rel-cur_sq_rel-cur_rms-cur_log_rms+cur_a1+cur_a2+cur_a3
        if best_depth < computed_depth:
            abs_rel = cur_abs_rel
            sq_rel = cur_sq_rel
            rms = cur_rms
            log_rms = cur_log_rms
            a1 = cur_a1
            a2 = cur_a2
            a3 = cur_a3
            best_depth = computed_depth
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
        print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))


    record_result(f'Current mIoU:{computed_miou}\n')
    record_result(f'Current model: {i_iter}\n')
    record_result(f'Current best mIoU:{cur_best_miou}\n')
    record_result(f'Current best model:{cur_best_model}\n')
    print("inters_over_union_classes[4]",inters_over_union_classes[4])

    display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)

    if depth_esti:


        record_result("cur_abs_rel', 'cur_sq_rel', 'cur_rms', 'cur_log_rms', 'cur_a1', 'cur_a2', 'cur_a3'")
        record_result( f"{cur_abs_rel, cur_sq_rel, cur_rms, cur_log_rms, cur_a1, cur_a2, cur_a3}")
        record_result("best_abs_rel', 'best_sq_rel', 'best_rms', 'best_log_rms', 'best_a1', 'best_a2', 'best_a3'")
        record_result(f"{abs_rel, sq_rel, rms, log_rms, a1, a2, a3}")



    return cur_best_miou, cur_best_model ,computed_miou,abs_rel,sq_rel,rms,log_rms,a1,a2,a3,best_depth



