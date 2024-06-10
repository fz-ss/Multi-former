import warnings
import os.path as osp
import torch.cuda
from torch.utils import data
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import argparse
import sys
# sys.path.append("/home/ailab/ailab/SYN/Trans_depth/Trans_depth3")
sys.path.append("/media/ailab/data/syn/Trans_depth3")
from depth_distribution.main.utils.dataset_util import compute_errors

from depth_distribution.main.domain_adaptation.eval_UDA1 import evaluate_domain_adaptation
from depth_distribution.main.dataset.cityscapes_isl import CityscapesDataSet
from depth_distribution.main.dataset.mapillary import MapillaryDataSet_1
from depth_distribution.main.model.build import build_feature_extractor, build_classifier, build_adversarial_discriminator_bin,build_adversarial_discriminator_cls


warnings.filterwarnings("ignore")
        
def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def main(args):

    expid = args.expid
    if expid == 1:
        from depth_distribution.configs.synthia_to_cityscapes_16cls  import cfg
    elif expid == 2:
        from depth_distribution.configs.synthia_to_cityscapes_7cls  import cfg
    elif expid == 3:
        from depth_distribution.configs.synthia_to_cityscapes_7cls_small  import cfg
    elif expid == 4:
        from depth_distribution.configs.synthia_to_mapillary_7cls  import cfg
    elif expid == 5:
        from depth_distribution.configs.synthia_to_mapillary_7cls_small  import cfg


    # load models
    # models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == "best":
        assert n_models == 1, "Not yet supported"
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == "Swin_S":
            device = cfg.GPU_ID
            feature_extractor = build_feature_extractor(cfg)
            feature_extractor.to(device)
            classifier,_ = build_classifier(cfg)
            classifier.to(device)
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        # models.append(model)

    # dataloaders
    fixed_test_size = True
    if cfg.TARGET == 'Cityscapes':
        test_dataset = CityscapesDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TEST.IMG_MEAN,
            labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
        )
        # test_dataset = CityscapesDataSet(  #conclude depth
        #     root=cfg.DATA_DIRECTORY_TARGET,
        #     list_path=cfg.DATA_LIST_TARGET,
        #     set=cfg.TRAIN.SET_TARGET,   #cfg.TRAIN.SET_TARGET,
        #     info_path=cfg.TRAIN.INFO_TARGET,
        #     crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
        #     mean=cfg.TEST.IMG_MEAN,
        #     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
        # )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    # eval
    if not osp.exists(args.pret_model):
        print('---Model does not exist!---')
        return
    print("Evaluating model", args.pret_model)

    checkpoint = torch.load(args.pret_model, map_location=torch.device('cpu'))
    feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
    feature_extractor.load_state_dict(feature_extractor_weights, strict = False)
    classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
    classifier.load_state_dict(classifier_weights)
    if "iteration" in checkpoint:
        iteration = checkpoint['iteration']
        print(iteration)
    # eval
    # hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = iter(test_loader)
    num_samples = len(test_loader)
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    MAX_DEPTH = 80 #50
    MIN_DEPTH = 1e-3
    for ind in tqdm(range(len(test_loader))):
        image, label, gt_depth,_,_ = next(test_iter)
        # s_ = next(test_iter)
        # print(len(s_))

        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            _ ,pred_depth,_= classifier(feature_extractor(image.cuda(device)))  #1,7,80,160
            interp = nn.Upsample(size=(gt_depth.shape[1], gt_depth.shape[2]), mode='bilinear', align_corners=True)
            pred_depth = interp(pred_depth)
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
            # pred = interp(pred_main)
            # pred = resize(   #1,7,1024,2048
            #     input=pred_main,
            #     size=label.shape[-2:],
            #     mode='bilinear',
            #     align_corners=False)
            abs_rel[ind], sq_rel[ind], rms[ind], log_rms[ind], a1[ind], a2[ind], a3[ind] = compute_errors(gt_depth[mask], pred_depth[mask])
        
        # save
            # pred_img = Image.fromarray(pred_depth.astype(np.int32)*100, 'I')
            # pred_img.save('%s/%05d_pred.png'%(save_dir, ind))
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    '''expid
    Syn_City_16cls ---> 1
    Syn_City_7cls ---> 2
    Syn_City_7cls_small ---> 3
    Syn_Map_7cls ---> 4
    Syn_Map_7cls_small ---> 5
    '''
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--expid', type=int, default=3, help='experiment id')
    parser.add_argument("--pret-model", type=str, default='/media/ailab/data/syn/Trans_depth3/depth_distribution/experiments/best/Syn3City/17500.pth', help="pretrained weights to be used for test")
    # parser.add_argument("--pret-model",type=str,default="/home/ailab/ailab/SYN/Trans_depth/Trans_depth3/depth_distribution/experiments/snapshots/SYNTHIA3Cityscapes_DeepLabv2_Depdis/6.pth",)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    print(torch.cuda.is_available())
    print("Called with args:")
    print(args)
    main(args)
