# -*-coding:utf-8-*-
import os.path as osp
import numpy as np
from easydict import EasyDict
from depth_distribution.main.utils import project_root

cfg = EasyDict()

# source domain
cfg.SOURCE = "SYNTHIA"

# target domain
cfg.TARGET = "Cityscapes"

# Number of workers for dataloading
cfg.NUM_WORKERS = 4

#pseudo labels number
cfg.MAX_ITERS_PSEUDO = 2975
#self_training number
cfg.MAX_ITERS_SELFTRAIN = 30000

# List of training images
cfg.DATA_LIST_SOURCE = str(project_root / "main/dataset/synthia_list/{}.txt")
cfg.DATA_LIST_TARGET = str(project_root / "main/dataset/cityscapes_list/{}.txt")

#Data Directories
cfg.DATA_DIRECTORY_SOURCE = "/media/ailab/data/yy/data/RAND_CITYSCAPES"
cfg.DATA_DIRECTORY_TARGET = "/media/ailab/data/yy/data/CityScapes"
# cfg.DATA_DIRECTORY_SOURCE = "/home/ailab/ailab/SYN/TransDA/datasets/synthia/RAND_CITYSCAPES"
# cfg.DATA_DIRECTORY_TARGET = "/home/ailab/ailab/SYN/TransDA/datasets/Cityscapes"

# Number of object classes
cfg.NUM_CLASSES = 7
cfg.USE_DEPTH = True
# cfg.USE_DEPTH = False

# Exp dirs
cfg.EXP_NAME = "SYNTHIA3Cityscapes"
cfg.EXP_ROOT = project_root / "experiments"
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, "snapshots")

# CUDA
cfg.GPU_ID = 1

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = "all"
cfg.TRAIN.SET_TARGET = "train"
cfg.TRAIN.SET_TARGET_SEL = "train"
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (640, 320)
cfg.TRAIN.INPUT_SIZE_TARGET = (640, 320)
cfg.TRAIN.INPUT_SIZE_TARGET_DIS = (640, 320)

# Class info
cfg.TRAIN.INFO_SOURCE = ""
cfg.TRAIN.INFO_TARGET = str(project_root / "main/dataset/cityscapes_list/info7class.json")

# Segmentation network params
cfg.TRAIN.MODEL = "Swin_S"

# cfg.TRAIN.MULTI_LEVEL = False  # in DADA paper we turn off this feature
cfg.TRAIN.RESTORE_FROM = "../../pretrained_models/DeepLab_resnet_pretrained_imagenet.pth"
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9

#Loss weight
cfg.TRAIN.LAMBDA_SEG_SRC = 1.0  # weight of source seg loss
# cfg.TRAIN.LAMBDA_BAL_SRC = 0.01  # weight of source balance loss
cfg.TRAIN.LAMBDA_DEP_SRC = 0.001 # weight of source depth loss
# cfg.TRAIN.LAMBDA_DEP_SRC = 1
cfg.TRAIN.LAMBDA_ADV_TAR = 1  # weight of target adv loss
# cfg.TRAIN.LAMBDA_BAL_TAR = 0.05 # weight of target balance loss

# Domain adaptation
cfg.TRAIN.DA_METHOD = "Depdis"

# Adversarial training params
cfg.TRAIN.LEARNING_RATE_D = 1e-3

# Other params
cfg.TRAIN.MAX_ITERS = 90000
cfg.TRAIN.EARLY_STOP = 90000
cfg.TRAIN.SAVE_PRED_EVERY = 500
cfg.TRAIN.SAVE_PRED_EVERY_SELFTRAIN =1000
cfg.TRAIN.SNAPSHOT_DIR = ""
cfg.TRAIN.RANDOM_SEED = 1234

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = "best"

# model
cfg.TEST.MODEL = ("Swin_S",)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# Test sets
cfg.TEST.SET_TARGET = "val"
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (640, 320)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / "main/dataset/cityscapes_list/info7class.json")

cfg.WEIGHTS="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"

cfg.SOLVER = EasyDict()
cfg.SOLVER.BASE_LR = 0.00006
cfg.SOLVER.BASE_LR_D = 0.0001
cfg.SOLVER.DIS = 'binary'
cfg.SOLVER.LR_METHOD = 'poly'
cfg.SOLVER.LR_POWER = 0.9

cfg.SOLVER.MOMENTUM = 0.9

cfg.SOLVER.WEIGHT_DECAY = 0.0005
cfg.SOLVER.WEIGHT_DECAY_BIAS = 0