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
import albumentations as A
def BGR2RGB(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def read_resize_video(filename, uplimit=4000, frame_shape=(256, 256)):
    frames = []
    cap = cv2.VideoCapture(filename)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame  = BGR2RGB(frame) 
            frames.append(frame)
        else:
            break
        cnt += 1
        if cnt >= uplimit:
            break
    cap.release()
    assert len(frames) > 0, f'{filename}: video with no frames!'
    return frames
def read_video(name, frame_shape,uplimit=4000):
    if name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(read_resize_video(name, uplimit=uplimit , frame_shape=(frame_shape[0],frame_shape[1])))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = video
    else:
        raise Exception("Unknown file extensions  %s" % name)
    return video_array
class FramesDataset(Dataset):
    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, augmentation_params=None, sample_percent=1.0):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.videos = [os.path.join(root_dir, each) for each in self.videos]
        self.random_seed = random_seed
        random.seed(random_seed)
        self.videos = random.sample(self.videos, round(len(self.videos)*sample_percent))
        self.frame_shape = tuple(frame_shape)
        self.id_sampling = id_sampling
        self.is_train = is_train
        self.identity_num = len(self.videos)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __len__(self):
        return len(self.videos)
    def __getitem__(self, idx):
        if not self.is_train:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            another_idx = (idx+13)%len(self.videos)
            name_another = self.videos[another_idx]
        else:
            another_idx = random.randint(0, self.identity_num-1)
            name_another = self.videos[another_idx]
        name = self.videos[idx]
        try:
            path = np.random.choice(glob.glob(os.path.join(name, '*.mp4')))
            path_another = np.random.choice(glob.glob(os.path.join(name_another, '*.mp4')))
        except:
            return self.__getitem__(random.randint(0, len(self.videos)-1))
        video_name = os.path.basename(path)
        video_another_name = os.path.basename(path_another)
        video_array = read_video(path, frame_shape=self.frame_shape)
        video_another_array = read_video(path_another, frame_shape=self.frame_shape)
        num_frames = len(video_array)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) 
        video_array = video_array[frame_idx]
        num_frames = len(video_another_array)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=1)) 
        video_another_array = video_another_array[frame_idx]
        out = {}
        source = Image.fromarray(np.array(video_array[0]))
        driving = Image.fromarray(np.array(video_array[1]))
        other_identity = Image.fromarray(np.array(video_another_array[0]))
        out['driving'] = self.transform(driving)
        out['source'] = self.transform(source)
        out['other_identity'] = self.transform(other_identity)
        out['name'] = video_name
        return out
class SpecifyInferenceDataset(Dataset):
    def __init__(self, root_dir, 
                       frame_shape=(256, 256, 3), 
                       specify_driving_name=[], 
                       specify_source_image=None, 
                       uplimit=4000
                       ):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.videos = [os.path.join(root_dir, each) for each in self.videos]
        self.specify_source_image = specify_source_image
        self.uplimit = uplimit
        if len(specify_driving_name) > 0:
            self.specify_videos = []
            for each_video in self.videos:
                for each_ in specify_driving_name:
                    if each_ in each_video:
                        self.specify_videos.append(each_video)
                        break
            self.videos = self.specify_videos
        assert len(self.videos)!=0, f"there is no any specify_video in {self.root_dir}"
        self.frame_shape = tuple(frame_shape)
        self.identity_num = len(self.videos)
        self.source = Image.fromarray(imageio.imread(self.specify_source_image))
        self.driving_image = []
        for video in self.videos:
            self.driving_image.extend(read_video(video, uplimit=uplimit, frame_shape=self.frame_shape))
        print(f"driving_image_num: {len(self.driving_image)}")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __len__(self):
        return len(self.driving_image)
    def __getitem__(self, idx):
        out = {}
        out['driving'] = self.transform(Image.fromarray(self.driving_image[0]))
        out['source'] = self.transform(self.source)
        self.driving_image = self.driving_image[1:]
        out['name'] = str(idx)
        return out
class SpecifyOneImgOneDirDataset(Dataset):
    def __init__(self, specify_source_image,
                       specify_driving_dir, 
                       frame_shape=(256, 256, 3), 
                       percent=1.0,
                       ):
        self.images = os.listdir(specify_driving_dir)
        temp_images = [each for each in self.images if each.endswith('.png')]
        try:
            if '_D' in temp_images[0]:
                self.images = sorted(temp_images, key=lambda s: int(s[:s.rfind('_')]))
            else:
                self.images = sorted(temp_images, key=lambda s: int(s[s.rfind('_')+1:-4])) 
                self.ori_idx = sorted([int(each[each.rfind('_')+1:-4]) for each in temp_images])
        except:
            self.images = sorted(temp_images, key=lambda s: int(s[s.rfind('/')+1:s.rfind('.')])) 
            self.ori_idx = sorted([int(each[each.rfind('/')+1:each.rfind('.')]) for each in temp_images])
        self.images = [os.path.join(specify_driving_dir, each) for each in self.images]
        self.images = self.images[:round(len(self.images)*percent)]
        self.specify_source_image = specify_source_image
        self.identity_num = len(self.images)
        self.source = Image.fromarray(imageio.imread(self.specify_source_image))
        self.driving_image = []
        self.driving_image_filename = []
        for img in self.images:
            self.driving_image_filename.append(img)
            self.driving_image.append(Image.fromarray(imageio.imread(img)))
        print(f"driving_image_num: {len(self.driving_image)}")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.length = len(self.driving_image)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        out = {}
        out['driving'] = self.transform(self.driving_image[idx])
        out['source'] = self.transform(self.source)
        out['name'] = os.path.basename(self.driving_image_filename[idx])
        return out
class SpecifyImageDataset(Dataset):
    def __init__(self, root_dir, key_word='_output'):
        self.root_dir = root_dir
        self.img_files = os.listdir(root_dir)
        predix_dict = {}
        for each_file in self.img_files:
            if key_word not in each_file:
                continue
            prefix = each_file[:each_file.find('_output')]
            if prefix not in predix_dict:
                predix_dict[prefix] = [each_file]
            else:
                predix_dict[prefix].append(each_file)
        for each_key in predix_dict:
            assert len(predix_dict[each_key]) == 2, "one identity must only have two images in speicify dir"
        self.source_imgs = []
        self.driving_imgs = []
        for each_key in predix_dict:
            self.source_imgs.extend(predix_dict[each_key])
            self.driving_imgs.extend(predix_dict[each_key][::-1])
        self.source_imgs = [os.path.join(root_dir, each) for each in self.source_imgs]
        self.driving_imgs = [os.path.join(root_dir, each) for each in self.driving_imgs]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __len__(self):
        return len(self.driving_imgs)
    def __getitem__(self, idx):
        out = {}
        out['driving'] = self.transform(Image.fromarray(imageio.imread(self.driving_imgs[idx])))
        out['source'] = self.transform(Image.fromarray(imageio.imread(self.source_imgs[idx])))
        out['name'] = str(idx)
        return out
def get_img_list(root_dir, identity2list, key_word='_output', recursive=True):
    files = os.listdir(root_dir)
    for each_file in tqdm.tqdm(files):
        if 'fixed_interval' in each_file:
            continue
        if 'foreground_mask' in each_file:
            continue
        real_path = os.path.join(root_dir, each_file)
        if os.path.isfile(real_path):
            if '.png' not in each_file:
                continue
            if key_word not in each_file:
                continue
            identity = os.path.basename(root_dir)
            if identity not in identity2list:
                identity2list[identity] = [real_path]
            else:
                identity2list[identity].append(real_path)
        elif os.path.isdir(real_path):
            if recursive:
                get_img_list(real_path, identity2list, key_word=key_word)
    return identity2list

class ImgsDataset(Dataset):
    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, augmentation_params=None, sample_percent=1.0, key_word='_output', load_mask=False,
                 color_aug=False, load_e4e_latent=False):
        self.root_dir = root_dir
        self.imgs = {}
        self.imgs = get_img_list(root_dir, self.imgs)
        self.identity_list = list(self.imgs.keys())
        self.load_mask = load_mask
        random.seed(random_seed)
        self.identity_list = random.sample(self.identity_list, round(len(self.identity_list)*sample_percent))
        self.identity_record_for_each_img = []
        for identity_key in self.identity_list:
            for each_img in self.imgs[identity_key]:
                self.identity_record_for_each_img.append((each_img, identity_key))
        self.frame_shape = tuple(frame_shape)
        self.id_sampling = id_sampling
        self.is_train = is_train
        self.identity_num = len(self.imgs)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.load_e4e_latent = load_e4e_latent
        self.color_aug = color_aug
        self.color_transform = [A.ToGray(p=0.5), A.RGBShift(p=0.5), A.ColorJitter(p=0.5), A.RandomBrightnessContrast(p=0.5)]
    def __len__(self):
        return len(self.identity_record_for_each_img)
    def __getitem__(self, idx):
        if not self.is_train:
            random.seed(idx)
        name = self.identity_record_for_each_img[idx][1]
        sampled_img_name = self.identity_record_for_each_img[idx][0]
        name_another = random.randint(0, len(self.identity_list)-1)
        name_another = self.identity_list[name_another]
        img_name = os.path.basename(name)
        img_another_name = os.path.basename(name_another)
        try:
            assert name!=name_another
            img_path = random.sample(self.imgs[name], 2)
            if sampled_img_name not in img_path: 
                img_path[0] = sampled_img_name
            img_another_path = random.sample(self.imgs[name_another], 1)
            out = {}
            source = Image.open(img_path[0])
            driving = Image.open(img_path[1])
            other_identity = Image.open(img_another_path[0])
            if self.color_aug:
                color_source = np.array(source)
                color_transform = random.choice(self.color_transform)
                color_source = color_transform(image=color_source)["image"]
                color_source = Image.fromarray(color_source)
                out['color_source'] = self.transform(color_source)
            if self.load_mask:
                source_mask = np.load(img_path[0][:-4] + '_warping_region.npy')
                driving_mask = np.load(img_path[1][:-4] + '_warping_region.npy')
                source_mask = source_mask.astype(np.float32)
                source_mask= cv2.resize(source_mask, (256,256), interpolation=cv2.INTER_NEAREST)
                driving_mask = driving_mask.astype(np.float32)
                driving_mask= cv2.resize(driving_mask, (256,256), interpolation=cv2.INTER_NEAREST)
            if self.load_e4e_latent:
                out['source_e4e_latent'] = torch.load(img_path[0][:-4]+'_e4e-latent.pth', 'cpu')
        except Exception as result:
            print("exception, ", result)
            print(f"load img error, where number: {idx}, name: {sampled_img_name}")
            return self.__getitem__(random.randint(0, len(self.identity_record_for_each_img)-1))
        out['driving'] = self.transform(driving)
        out['source'] = self.transform(source)
        if self.load_mask:
            out['driving_mask'] = driving_mask
            out['source_mask'] = source_mask
        out['other_identity'] = self.transform(other_identity)
        out['name'] = img_name
        return out

class ConcatDataset(Dataset):
    def __init__(self, dataset_list, sample_mode='normal', sample_ratio=None):
        self.dataset_list = dataset_list
        self.sample_mode = sample_mode
        self.sample_ratio = sample_ratio
        self.each_dataset_length = []
        for each_dataset in dataset_list:
            self.each_dataset_length.append(len(each_dataset))
        if sample_mode == 'normal':
            self.cum_sum = [0]
            for i in range(len(dataset_list)):
                self.cum_sum.append(self.cum_sum[-1]+len(dataset_list[i]))
        elif sample_mode == 'average':
            self.cum_sum
        elif sample_mode == 'custom':
            pass
        else:
            raise NotImplementedError
        self.sub_dataset_num = len(dataset_list)
    def __len__(self):
        return sum(self.each_dataset_length)
    def __getitem__(self, idx):
        if self.sample_mode == 'normal':
            for i in range(self.sub_dataset_num):
                if idx < self.cum_sum[i]:
                    return self.dataset_list[i][idx-self.cum_sum[i-1]] 
        else:
            pass
