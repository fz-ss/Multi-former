import numpy as np
import os
import cv2

from depth_distribution.main.utils.serialization import json_load
from depth_distribution.main.dataset.base_dataset import BaseDataset



class CityscapesDataSet(BaseDataset):  #target
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=None, labels_size=None, iternum=0):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
        self.realbeginNum = 0
        self.iternum = iternum
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str_)
        self.mapping = np.array(self.info['label2train'], dtype=np.int64)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        if self.iternum > 0 and (self.realbeginNum + 5) < self.iternum:
            self.realbeginNum += 1
            return 1, 2, 3, 4, 5

        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        imagepath = str(img_file)
        image = self.preprocess(image)
        return image.copy(), label, np.array(image.shape), name, imagepath



class CityscapesDataSet_1(BaseDataset):  #test
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=None, labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str_)
        self.mapping = np.array(self.info['label2train'], dtype=np.int64)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
            img_file, label_file, name = self.files[index]
            label = self.get_labels(label_file)
            label = self.map_labels(label).copy()
            image = self.get_image(img_file)
            image = self.preprocess(image)
            return image.copy(), label, np.array(image.shape), str(img_file),str(img_file)



class CityscapesDataSet_2(BaseDataset): #self_train
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=None, labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str_)
        self.mapping = np.array(self.info['label2train'], dtype=np.int64)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name

        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def get_pseudo(self, file):
        label_pseudo = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.uint8)
        return label_pseudo
    def get_depth(self,name):
        label_name = name.replace("leftImg8bit", "disparity")
        disp_img_file = self.root / 'disparity' / self.set / label_name
        depth = cv2.imread(disp_img_file, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = cv2.resize(depth, interpolation=cv2.INTER_NEAREST)
        depth = 65536.0 / (depth + 1)  # inverse depth
        return depth

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob) ##按照概率随机选择一个类
        f1 = np.random.choice(self.samples_with_class[c]) ##从类中选择一张图像;f1 is labeled image name
        index = f1.split('_')[0]
        return int(index)

    def __getitem__(self, index):
            # index = self.get_rare_class_sample()
            img_file, label_file, name = self.files[index]
            image = self.get_image(img_file)
            image = self.preprocess(image)
            # name1 = os.path.basename(img_file) + '.tiff'
            # label_file_pseudo = self.root / 'pseudo_labels' / name1
            # label_pseudo = self.get_pseudo(label_file_pseudo)
            # depth = ''
            depth = self.get_depth(name)
            
            return image.copy(), '', np.array(image.shape), name, depth


