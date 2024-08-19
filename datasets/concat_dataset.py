import os
import pdb
import math
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread
import imageio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import glob
import cv2
import torchvision.transforms as transforms
import random
from PIL import Image
import tqdm
import torch
import pickle
class ConcatDataset(Dataset):
    def __init__(self, datasets = [], proportion = []):
        self.datasets = datasets
        self.proportion = proportion
        assert sum(proportion) == 1
        self.prob_dis = np.cumsum(proportion)
        self.sub_dataset_num = len(datasets)
        self.sub_dataset_length = []
        for each_dataset in self.datasets:
            self.sub_dataset_length.append(len(each_dataset))
    def __len__(self):
        return np.cumsum(self.sub_dataset_length)[-1]
    def __getitem__(self, idx):
        value = random.random()
        for i in range(self.sub_dataset_num):
            if value < self.prob_dis[i]:
                idx = random.randint(0, self.sub_dataset_length[i] - 1)
                return self.datasets[i][idx]
