import csv
import os

import torch
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset


# Selected imagenet. The .csv file format:
# class_index, class, image_name
# 0,n01440764,ILSVRC2012_val_00002138.JPEG
# 2,n01484850,ILSVRC2012_val_00004329.JPEG
# ...
class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)
    def __len__(self):
        return len(self.selected_list)

class NumpyImages(Dataset):
    def __init__(self, npy_dir, transforms=None):
        super(NumpyImages, self).__init__()
        npy_ls = []
        for npy_name in os.listdir(npy_dir):
            if npy_name[:5] == 'batch':
                npy_ls.append(npy_name)
        self.data = []
        for npy_ind in range(len(npy_ls)):
            self.data.append(np.load(npy_dir + '/batch_{}.npy'.format(npy_ind)))
        self.data = np.concatenate(self.data, axis=0)
        self.target = np.load(npy_dir+"/labels.npy")
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float() / 255, self.target[index]
    
    def __len__(self,):
        return len(self.target)