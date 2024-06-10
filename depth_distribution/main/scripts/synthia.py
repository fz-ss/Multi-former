# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp

import cv2
import mmcv
import numpy as np
from PIL import Image


def convert_to_train_id(file):
    args = parse_args()
    # re-assign labels to match the format of Cityscapes
    # PIL does not work with the image format, but cv2 does
    label = cv2.imread(file, cv2.IMREAD_UNCHANGED)[:,-1]
    # mapping based on README.txt from SYNTHIA_RAND_CITYSCAPES
    if args.expid ==1:
        id_to_trainid = {
            3: 0,
            4: 1,
            2: 2,
            21: 3,
            5: 4,
            7: 5,
            15: 6,
            9: 7,
            6: 8,
            1: 9,
            10: 10,
            17: 11,
            8: 12,
            19: 13,
            12: 14,
            11: 15
    }
    if args.expid ==2:
        id_to_trainid = {
            1: 4,
            2: 1,
            3: 0,
            4: 0,
            5: 1,
            6: 3,
            7: 2,
            8: 6,
            9: 2,
            10: 5,
            11: 6,
            15: 2,
            22: 0}
    # label_copy = 255 * np.ones(label.shape, dtype=np.uint8) #全为255，即无标签
    sample_class_stats = {}
    for k, v in id_to_trainid.items(): #源域和目标域的标签
        k_mask = label == k  #源域中是第k类的像素，Ture or False
        # label_copy[k_mask] = v  #将这些像素点的标签转换为目标域的标签
        n = int(np.sum(k_mask)) #统计True的个数
        if n > 0:
            sample_class_stats[k] = n  #目标域的标签的像素个数
    # new_file = file.replace('.png', '_labelTrainIds.png')
    # assert file != new_file
    sample_class_stats['file'] = file
    # Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats  #目标域对应的标签某个类的个数


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SYNTHIA annotations to TrainIds')
    parser.add_argument('--synthia_path', default='/media/ailab/data/yy/data/RAND_CITYSCAPES',help='gta data path')
    parser.add_argument('--gt-dir', default='GT/LABELS', type=str)  #相对路径
    parser.add_argument('-o', '--out-dir', default="/media/ailab/data/syn/Trans_depth3/depth_distribution/data/Syn1City",help='output path,1 present 16 classes and full resolution;2 present 7 classes and full resolution')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    parser.add_argument('--expid',default=1, help = '1 present 16 classes and full resolution;2 present 7 classes and full resolution')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    synthia_path = args.synthia_path
    out_dir = args.out_dir if args.out_dir else synthia_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(synthia_path, args.gt_dir)  #gt的绝对路径

    poly_files = []
    for poly in mmcv.scandir(   #加载所有图像
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files) #按顺序排序

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
