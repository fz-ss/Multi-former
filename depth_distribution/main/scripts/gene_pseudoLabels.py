import argparse
import pprint
import warnings
import numpy as np
import torch
from torch.utils import data
from collections import OrderedDict
import os
import sys
# sys.path.append("../../../")
# sys.path.append('/media/ailab/data/syn/Trans_depth1')
# sys.path.append("/home/ailab/ailab/SYN/Trans_depth/Trans_depth2")
sys.path.append("/media/ailab/data/syn/Trans_depth3")
from depth_distribution.main.dataset.cityscapes import CityscapesDataSet_1
from depth_distribution.main.dataset.mapillary import MapillaryDataSet_1
from depth_distribution.main.model.build import build_feature_extractor, build_classifier, build_adversarial_discriminator_bin,build_adversarial_discriminator_cls
from depth_distribution.main.model.checkpoint import load_checkpoint
from depth_distribution.main.domain_adaptation.gene_UDA import gene_pseudo_labels_1, gene_pseudo_labels_2,\
gene_pseudo_labels_3, gene_pseudo_labels_4, gene_pseudo_labels_5


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

    print("Using config:")
    pprint.pprint(cfg)

    def _init_fn(worker_id):
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)


    if cfg.TRAIN.MODEL == "Swin_S":
        # SEGMNETATION NETWORK
        device = cfg.GPU_ID

        feature_extractor = build_feature_extractor(cfg)
        feature_extractor.to(device)

        classifier,_ = build_classifier(cfg)
        classifier.to(device)
        for name,param in classifier.named_parameters():
            print(f"Layer:{name},Size:{param.size()}")    #类似于参数的shape
    
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

    feature_extractor.eval()
    classifier.eval()

    if cfg.TARGET == 'Cityscapes':
        target_dataset = CityscapesDataSet_1(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET_SEL,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.MAX_ITERS_PSEUDO,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
        )
    elif cfg.TARGET == 'Mapillary':
        target_dataset = MapillaryDataSet_1(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET_SEL,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=cfg.MAX_ITERS_PSEUDO,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=True
        )
    else:
        raise NotImplementedError(f"Not yet supported dataset {cfg.TARGET}")

    target_loader = data.DataLoader(
        target_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    os.makedirs(args.output_dir,exist_ok = True)
    if expid == 1:
        gene_pseudo_labels_1(feature_extractor,classifier, target_loader, args.output_dir, cfg)
    elif expid == 2:
        gene_pseudo_labels_2(feature_extractor,classifier, target_loader, args.output_dir, cfg)
    elif expid == 3:
        gene_pseudo_labels_3(feature_extractor,classifier, target_loader, args.output_dir, cfg)
    elif expid == 4:
        gene_pseudo_labels_4(feature_extractor,classifier, target_loader, args.output_dir, cfg)
    elif expid == 5:
        gene_pseudo_labels_5(feature_extractor,classifier, target_loader, args.output_dir, cfg)

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
    parser.add_argument("--pret-model", type=str, default='/media/ailab/data/syn/Trans_depth3/depth_distribution/experiments/best/Syn1City/6500.pth', help="pretrained weights to be used for initialization")
    parser.add_argument('--output-dir', type=str,  default='/media/ailab/data/yy/data/CityScapes/pseudo_labels', help='folder where pseudo labels are stored')
    return parser.parse_args()

if __name__ == "__main__":
    main()