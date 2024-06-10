import argparse
import os
import os.path as osp
import pprint
import random
import warnings
from collections import OrderedDict
import numpy as np
import time, itertools
import torch
from torch.utils import data
import torch.optim as optim
import sys
# sys.path.append('../../../')
sys.path.append("/media/ailab/data/syn/Trans_depth3")
# sys.path.append("/home/ailab/ailab/SYN/Finally/Trans_depth3")
from depth_distribution.main.dataset.synthia import SYNTHIADataSetDepth
from depth_distribution.main.dataset.cityscapes import CityscapesDataSet_1, CityscapesDataSet_2
from depth_distribution.main.dataset.mapillary import MapillaryDataSet_1, MapillaryDataSet_2
from depth_distribution.main.model.build import build_feature_extractor, build_classifier, build_adversarial_discriminator_bin,build_adversarial_discriminator_cls
# from depth_distribution.main.scripts.self_depth import selftrain_depdis #深度估计
# from depth_distribution.main.domain_adaptation.selftrain_UDA import selftrain_depdis
from depth_distribution.main.domain_adaptation.pse_mask import selftrain_depdis  #统计PLW模块
from depth_distribution.main.dataset.cityscapes_isl import CityscapesDataSet

warnings.filterwarnings("ignore")
def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict
def main():
    # LOAD ARGS
    args = get_arguments()
    print("Called with args:")
    print(args)

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

    # auto-generate exp name if not specified
    if cfg.EXP_NAME == "":
        cfg.EXP_NAME = f"{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}"

    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == "":
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME, 'self_train_model')
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    print("Using config:")
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if cfg.TRAIN.MODEL == "Swin_S":
        # SEGMNETATION NETWORK
        feature_extractor = build_feature_extractor(cfg)
        device = cfg.GPU_ID

        feature_extractor = build_feature_extractor(cfg)
        feature_extractor.to(device)

        classifier ,aux = build_classifier(cfg)
        classifier.to(device)
    
        if args.pret_model != '':
            
            checkpoint = torch.load(args.pret_model, map_location=torch.device('cpu'))
            feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
            feature_extractor.load_state_dict(feature_extractor_weights, strict = False)
            classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
            classifier.load_state_dict(classifier_weights)
            if "iteration" in checkpoint:
                iteration = checkpoint['iteration']
                print(iteration)

        else:
            raise NotImplementedError(f"Not pret_model!")
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    feature_extractor.train()
    feature_extractor.to(cfg.GPU_ID)
    classifier.train()
    classifier.to(cfg.GPU_ID)
    aux.train()
    aux.to(cfg.GPU_ID)

    # feature_extractor's optimizer
    optimizer_fea = torch.optim.AdamW(
        feature_extractor.parameters(), 
        lr=cfg.SOLVER.BASE_LR, 
        betas=(0.9, 0.999), 
        weight_decay=0.01)
    # classifier's optimizer
    optimizer_cls = torch.optim.AdamW(
        itertools.chain(classifier.parameters(),aux.parameters()), 
        lr=cfg.SOLVER.BASE_LR, 
        betas=(0.9, 0.999), 
        weight_decay=0.01)
    # DATALOADERS
    source_dataset = SYNTHIADataSetDepth(
        root=cfg.DATA_DIRECTORY_SOURCE,
        list_path=cfg.DATA_LIST_SOURCE,
        set=cfg.TRAIN.SET_SOURCE,
        rcs_enabled = True,
        num_classes=cfg.NUM_CLASSES,
        max_iters=cfg.MAX_ITERS_SELFTRAIN,
        crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
        mean=cfg.TRAIN.IMG_MEAN,
        iternum=0,
        use_depth=cfg.USE_DEPTH,
        expid = expid
    )

    if cfg.TARGET == 'Cityscapes':
        # target_dataset = CityscapesDataSet_2(
        #     root=cfg.DATA_DIRECTORY_TARGET,
        #     list_path=cfg.DATA_LIST_TARGET,
        #     set=cfg.TRAIN.SET_TARGET,
        #     info_path=cfg.TRAIN.INFO_TARGET,
        #     max_iters=cfg.MAX_ITERS_SELFTRAIN,
        #     crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
        #     mean=cfg.TRAIN.IMG_MEAN
        # )
        target_dataset = CityscapesDataSet(  #conclude depth
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET,
            info_path=cfg.TRAIN.INFO_TARGET,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TEST.IMG_MEAN,
            labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
        )

    elif cfg.TARGET == 'Mapillary':
        target_dataset = MapillaryDataSet_2(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET_SEL,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.MAX_ITERS_SELFTRAIN,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=True
        )
    else:
        raise NotImplementedError(f"Not yet supported dataset {cfg.TARGET}")
    source_loader = data.DataLoader(
        source_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        # shuffle=(src_train_sampler is None),
        shuffle=True,
        pin_memory=True,
        sampler=None,
        worker_init_fn=_init_fn,
        # drop_last=True,
    )
    target_loader = data.DataLoader(
        target_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        # worker_init_fn=_init_fn,
        # drop_last=True,
    )

    if cfg.TARGET == 'Cityscapes':
        # test_dataset = CityscapesDataSet_1(
        #     root=cfg.DATA_DIRECTORY_TARGET,
        #     list_path=cfg.DATA_LIST_TARGET,
        #     set=cfg.TEST.SET_TARGET,
        #     info_path=cfg.TEST.INFO_TARGET,
        #     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
        #     mean=cfg.TEST.IMG_MEAN,
        #     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
        # )
        test_dataset = CityscapesDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TEST.IMG_MEAN,
            labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
        )
    elif cfg.TARGET == 'Mapillary':
        test_dataset = MapillaryDataSet_1(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=True
        )


    test_loader = data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    if args.depth_esti:
        selftrain_depdis(feature_extractor,classifier,aux, optimizer_fea,optimizer_cls,source_loader, target_loader,test_loader, cfg,args.depth_esti)
    else:
        selftrain_depdis(feature_extractor,classifier,aux, optimizer_fea,optimizer_cls,source_loader, target_loader,test_loader, cfg)

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation training")
    '''expid
    Syn_City_16cls ---> 1
    Syn_City_7cls ---> 2
    Syn_City_7cls_small ---> 3
    Syn_Map_7cls ---> 4
    Syn_Map_7cls_small ---> 5
    '''
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument("--random-train", action="store_true", help="not fixing random seed.")
    parser.add_argument("--depth_esti",type=bool,default=False,help="")
    parser.add_argument("--pret-model", type=str, default="/media/ailab/data/syn/Trans_depth3/depth_distribution/experiments/best/Syn1City/29399.pth", help="pretrained weights to be used for initialization")
    return parser.parse_args()

if __name__ == "__main__":
    main()