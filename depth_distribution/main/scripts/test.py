import warnings

import torch.cuda
from torch.utils import data
import argparse
import sys
# sys.path.append("/home/ailab/ailab/SYN/Trans_depth/Trans_depth2")
sys.path.append("/media/ailab/data/syn/Trans_depth3")

from depth_distribution.main.domain_adaptation.eval_UDA1 import evaluate_domain_adaptation
from depth_distribution.main.dataset.cityscapes import CityscapesDataSet_1
from depth_distribution.main.dataset.mapillary import MapillaryDataSet_1
from depth_distribution.main.dataset.cityscapes_isl import CityscapesDataSet
from depth_distribution.main.model.build import build_feature_extractor, build_classifier, build_adversarial_discriminator_bin,build_adversarial_discriminator_cls


warnings.filterwarnings("ignore")

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
        fixed_test_size = False

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    # eval
    evaluate_domain_adaptation(feature_extractor,classifier, test_loader, cfg, args.pret_model, fixed_test_size=fixed_test_size)


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
    parser.add_argument('--expid', type=int, default=4, help='experiment id')
    parser.add_argument("--pret-model", type=str, default='/media/ailab/data/syn/Trans_depth3/depth_distribution/experiments/snapshots/SYNTHIA1Mapillary/self_train_model/23499.pth', help="pretrained weights to be used for test")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    print(torch.cuda.is_available())
    print("Called with args:")
    print(args)
    main(args)
