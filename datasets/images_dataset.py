from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torch
import torchvision.transforms as transforms
import random
import pdb
class ImagesDataset(Dataset):
	def __init__(self, source_root, target_root, opts, dataset_percent=1.0, 
				 target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_paths = self.source_paths[:round(len(self.source_paths)*dataset_percent)]
		self.target_paths = self.target_paths[:round(len(self.target_paths)*dataset_percent)]
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		self.grid_transform = transforms.RandomPerspective(distortion_scale=opts.ffhq_distortion_scale, p=1.0)
	def __len__(self):
		return len(self.source_paths)
	def __getitem__(self, index):
		try:
			from_path = self.source_paths[index]
			from_im = Image.open(from_path)
			from_im = from_im.convert('RGB')
			index = random.randint(0, len(self.source_paths)-1)
			to_path = self.target_paths[index]
			to_im = Image.open(to_path).convert('RGB')
			if self.target_transform:
				to_im = self.target_transform(to_im)
			if self.source_transform:
				from_im = self.source_transform(from_im)
			else:
				from_im = self.target_transform(from_im)
			output = {}
			output['source'] = from_im
			if self.opts.ffhq_aug_rate > 0:
				output['driving'] = self.grid_transform(output['source'])
			else:
				output['driving'] = output['source']
			output['other_identity'] = to_im 
			output['source_e4e_latent'] = torch.load(from_path[:-4]+'_e4e-latent.pth', 'cpu')
			output['source_slim'] = from_im
			output['driving_slim'] = output['driving']
			output['other_identity'] = to_im
			output['name'] = from_path
		except Exception as result:
			print(result)
			return self.__getitem__(random.randint(0, len(self.source_paths)-1))
		return output
class ImagesDatasetV1(Dataset):
	def __init__(self, source_root, target_root, opts, dataset_percent=1.0, 
				 target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_paths = self.source_paths[:round(len(self.source_paths)*dataset_percent)]
		self.target_paths = self.target_paths[:round(len(self.target_paths)*dataset_percent)]
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		self.grid_transform = transforms.RandomPerspective(distortion_scale=opts.ffhq_distortion_scale, p=1.0)
	def __len__(self):
		return len(self.source_paths)
	def __getitem__(self, index):
		try:
			from_path = self.source_paths[index]
			from_im = Image.open(from_path)
			from_im = from_im.convert('RGB')
			index = random.randint(0, len(self.source_paths)-1)
			to_path = self.target_paths[index]
			to_im = Image.open(to_path).convert('RGB')
			if self.target_transform:
				to_im = self.target_transform(to_im)
			if self.source_transform:
				from_im = self.source_transform(from_im)
			else:
				from_im = self.target_transform(from_im)
			output = {}
			output['source'] = from_im
			if self.opts.ffhq_aug_rate > 0:
				output['driving'] = self.grid_transform(output['source'])
			else:
				output['driving'] = output['source']
			output['other_identity'] = to_im 
			output['source_e4e_latent'] = torch.load(from_path[:-4]+'_e4e-latent.pth', 'cpu')
			output['other_identity'] = to_im
			output['name'] = from_path
		except Exception as result:
			print(result)
			return self.__getitem__(random.randint(0, len(self.source_paths)-1))
		return output