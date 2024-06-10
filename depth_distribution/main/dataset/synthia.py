import numpy as np
import os
import json
import os.path as osp

import mmcv
import torch.nn

from depth_distribution.main.dataset.base_dataset import BaseDataset
import cv2
from depth_distribution.main.dataset.depth import get_depth
import math
def get_rcs_class_probs(data_root, temperature,expid):
    if expid ==1:
        with open(osp.join("/media/ailab/data/syn/Trans_depth3/depth_distribution/data/Syn1City", 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)
    else:
        with open(osp.join("/media/ailab/data/syn/Trans_depth3/depth_distribution/data/Syn2City", 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


class SYNTHIADataSetDepth(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        set="all",
        rcs_enabled = True,
        num_classes=16,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        iternum=0,
        use_depth=True,
        expid = 1
    ):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        self.realbeginNum = 0
        self.iternum = iternum
        self.expid = expid
        self.num_classes = num_classes
        self.rcs_enabled = rcs_enabled
        # map to cityscape's ids
        if num_classes == 16:
            self.id_to_trainid = {
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
                11: 15,
            }
        elif num_classes == 7:
            self.id_to_trainid = {
                1:4, 
                2:1, 
                3:0, 
                4:0, 
                5:1, 
                6:3, 
                7:2, 
                8:6, 
                9:2, 
                10:5, 
                11:6, 
                15:2, 
                22:0}
        else:
            raise NotImplementedError(f"Not yet supported {num_classes} classes")
        self.use_depth = use_depth
        if self.use_depth:
            for (i, file) in enumerate(self.files):
                img_file, label_file, name = file
                # density_file = self.root / "source_density_maps" / name
                depth_file_value = self.root / "Depth" / name
                self.files[i] = (img_file, label_file,  depth_file_value, name)
            # disable multi-threading in opencv. Could be ignored
            import os
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            if self.rcs_enabled:
                self.rcs_class_temp =0.05 #rcs_cfg['class_temp'] #0.01
                self.rcm_class_temp = 2
                self.rcs_min_crop_ratio = 0.5 #rcs_cfg['min_crop_ratio'] #0.5
                self.rcs_min_pixels = 3000 #rcs_cfg['min_pixels'] #3000

                self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                    root, self.rcs_class_temp,self.expid)
                self.rcm_classes, self.rcm_classprob = get_rcs_class_probs(
                    root, self.rcm_class_temp,self.expid)
                mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
                mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')
                self.prob_target = np.ones(self.num_classes)
                for k, v in self.id_to_trainid.items():
                    if self.prob_target[v] == 1:
                        self.prob_target[v] = self.rcm_classprob[self.rcm_classes.index(k)]
                    else:
                        self.prob_target[v] = self.prob_target[v]+self.rcm_classprob[self.rcm_classes.index(k)]
                if expid==1:
                    with open(
                          osp.join("/media/ailab/data/syn/Trans_depth3/depth_distribution/data/Syn1City",
                                   'samples_with_class.json'), 'r') as of:
                         samples_with_class_and_n = json.load(of)
                else:
                    with open(
                          osp.join("/media/ailab/data/syn/Trans_depth3/depth_distribution/data/Syn2City",
                                   'samples_with_class.json'), 'r') as of:
                         samples_with_class_and_n = json.load(of)
                    
                samples_with_class_and_n = {
                    int(k): v
                    for k, v in samples_with_class_and_n.items()
                    if int(k) in self.rcs_classes
                }
                self.samples_with_class = {}
                for c in self.rcs_classes:
                    self.samples_with_class[c] = []
                    for file, pixels in samples_with_class_and_n[c]:
                        if pixels > self.rcs_min_pixels:
                            self.samples_with_class[c].append(file.split('\\')[-1])
                    assert len(self.samples_with_class[c]) > 0

    def get_metadata(self, name):
        img_file = self.root / "RGB" / name
        label_file = self.root / "parsed_LABELS" / name
        return img_file, label_file

    def get_gaosipro(self, name):
        name1 = os.path.basename(name).replace('.png', '')
        name2 = os.path.dirname(name)
        xadd = None
        for i in range(self.num_classes):
            name = name2 + os.sep + name1 + '-' + str(i) + '.tiff'
            depthPro = cv2.imread(name, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
            x = np.expand_dims(depthPro, 0)
            if i == 0:
                xadd = x
            else:
                xadd = np.append(xadd, x, axis=0)
        return xadd
    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob) ##按照概率随机选择一个类
        f1 = np.random.choice(self.samples_with_class[c]) ##从类中选择一张图像;f1 is labeled image name
        index = f1.split('.')[0]

        return int(index)



    def __getitem__(self, index):
        if self.iternum > 0 and (self.realbeginNum + 5) < self.iternum:
            self.realbeginNum += 1
            return 1, 2, 3, 4, 5, 6

        if self.use_depth:
            if self.rcs_enabled:

                index  = self.get_rare_class_sample()
            
            img_file, label_file,depth_file_value, name = self.files[index]


        else:
            img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        if self.use_depth:
                #mapping
                # density_pre_source = self.get_gaosipro(density_file)* 1e6
                # density_pre_source = (1 - np.exp(-density_pre_source))* 255
                depthvalue = self.get_depth(depth_file_value)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)

        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
            # if prob_target[v] == 1:
            #     prob_target[v] = self.rcs_classprob[self.rcs_classes.index(k)]
            # else:
            #     prob_target[v] = prob_target[v]+self.rcs_classprob[k]

        image = self.preprocess(image)
        image = image.copy()
        label_copy = label_copy.copy()
        shape = np.array(image.shape)
        if self.use_depth:
            return image, label_copy, depthvalue.copy(), shape, name,self.prob_target

    def get_depth(self, file):
        return get_depth(self, file)
