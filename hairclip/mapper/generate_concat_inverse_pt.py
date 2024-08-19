import os
import pdb
import tqdm
import torch
target_dir = '/root/data/.shared_8t/qing_chang/encoder4editing-main/workdirs/second-stage-complete-bs8_mannual_filter_data_robust-official-parameter-pretrain/robust-official-parameter_fixed-lr/20K_HFGI_v3_10_2/official-parameter-stage2/checkpoints/60K/55_as_source/other_driven-use-resize_random-noise-false/driving_file/'
saved_path_path = 'expressive_select_e4e_inverse.pth'
files = os.listdir(target_dir)
files = [each for each in files if each.endswith('_e4e-latent.pth')]
files = [os.path.join(target_dir, each) for each in files]
save_pth_data = []
for each_pth in tqdm.tqdm(files):
	data = torch.load(each_pth, 'cpu')
	save_pth_data.append(data)
save_pth_data = torch.cat(save_pth_data, dim=0)
torch.save(save_pth_data, saved_path_path)