import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from depth_distribution.main.utils.misc import resize
import os
from depth_distribution.main.utils.visualization import save_image

from depth_distribution.main.utils.func import per_class_iu, fast_hist
from depth_distribution.main.utils.serialization import pickle_dump, pickle_load


def evaluate_domain_adaptation( feature_extractor,classifier, test_loader, cfg, restore_from,fixed_test_size=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'best':
        eval_best(cfg, feature_extractor,classifier, device, test_loader, interp, fixed_test_size, restore_from)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict['model'])
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
        
def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def eval_best(cfg, feature_extractor,classifier,
              device, test_loader, interp,
              fixed_test_size, restore_from):
    # assert len(models) == 1, 'Not yet supported multi models in this mode'

    cur_best_miou = -1
    cur_best_model = ''

    if not osp.exists(restore_from):
        print('---Model does not exist!---')
        return
    print("Evaluating model", restore_from)

    checkpoint = torch.load(restore_from, map_location=torch.device('cpu'))
    del_keys = ["enc5_1.weight","enc5_1.bias","enc5_2.weight","enc5_2.bias","enc5_3.weight","enc5_3.bias"]
    feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
    feature_extractor.load_state_dict(feature_extractor_weights, strict = False)
    classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
    for k in del_keys:
        del classifier_weights[k]
    classifier.load_state_dict(classifier_weights,strict=False)
    # if "iteration" in checkpoint:
    #     iteration = checkpoint['iteration']
    #     print(iteration)

    # feature_extractor.init_weights()
    # classifier.init_weights()
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = iter(test_loader)
    save_dir = "/media/ailab/data/syn/data/Multi_former"
    for index in tqdm(range(len(test_loader))):
        image, label, _, name,_ = next(test_iter)
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            pred_main ,_,_= classifier(feature_extractor(image.cuda(device)))  #1,7,80,160
            pred = interp(pred_main)
            # pred = resize(   #1,7,1024,2048
            #     input=pred_main,
            #     size=label.shape[-2:],
            #     mode='bilinear',
            #     align_corners=False)
            output = pred.max(1)[1]  #1,1024,2048
            output=output.cpu()
            output = output.numpy()[0]
        label = label.numpy()[0]
        ######test 4 alone
        label_save = label
        label_save = np.squeeze(label_save)
        ######test 1 alone over
        image = image.numpy()[0]
        image = torch.from_numpy(image)

        out_dir = os.path.join(save_dir, "Mapillary_pred(73)")
        save_image(out_dir, image, None, f'{index}_0img')
        save_image(out_dir, output, None, f'{index}_1pred')
        save_image(out_dir, label_save, None, f'{index}_2gt')
        
        
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        if index > 0 and index % 100 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(
                index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
    inters_over_union_classes = per_class_iu(hist)

    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    if cur_best_miou < computed_miou:
        cur_best_miou = computed_miou
        cur_best_model = restore_from
    print('\tCurrent mIoU:', computed_miou)
    print('\tCurrent best model:', cur_best_model)
    print('\tCurrent best mIoU:', cur_best_miou)
    display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)



