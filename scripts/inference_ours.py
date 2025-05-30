"""
This file runs the main training/val loop
"""
import os
import copy
import json
import tqdm
import math
import sys
import pprint
import torch
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_first_stage_complete_dp import Coach

def main():
	opts = TrainOptions().parse()
	#temp_opts = copy.deepcopy(temp_opts)
	previous_train_ckpt = None
	if opts.resume_training_from_ckpt:
		opts, previous_train_ckpt = load_train_checkpoint(opts)
	else:
		setup_progressive_steps(opts)
		create_initial_experiment_dir(opts)

	coach = Coach(opts, previous_train_ckpt)

	coach.inference(resize=False)

def load_train_checkpoint(opts):
	train_ckpt_path = opts.resume_training_from_ckpt
	previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
	new_opts_dict = vars(opts)

	previous_train_ckpt['opts']['dataset_type'] = opts.dataset_type 
	previous_train_ckpt['opts']['test_batch_size'] = opts.test_batch_size 
	previous_train_ckpt['opts']['progressive_steps'] = ''
	#previous_train_ckpt['opts']['learning_rate'] = opts.learning_rate
	previous_train_ckpt['opts']['exp_dir'] = opts.exp_dir
	#previous_train_ckpt['opts']['max_steps'] = opts.max_steps
	if 'driving_image_dir' in previous_train_ckpt['opts']:
		previous_train_ckpt['opts']['driving_image_dir'] = opts.driving_image_dir
	if 'source_image' in previous_train_ckpt['opts']:
		previous_train_ckpt['opts']['source_image'] = opts.source_image

	opts = previous_train_ckpt['opts'] 
	opts['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(opts, new_opts_dict)
	pprint.pprint(opts)
	opts = Namespace(**opts)
	if opts.sub_exp_dir is not None:
		sub_exp_dir = opts.sub_exp_dir
		opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
		create_initial_experiment_dir(opts)
	return opts, previous_train_ckpt

def setup_progressive_steps(opts):
	log_size = int(math.log(opts.stylegan_size, 2))
	num_style_layers = 2*log_size - 2
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

	assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
		"Invalid progressive training input"

def is_valid_progressive_steps(opts, num_style_layers):
	return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts):
	if os.path.exists(opts.exp_dir):
		print("exp dir is alread created before, be careful")
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
	for k, v in new_opts.items():
		if k not in ckpt_opts:
			ckpt_opts[k] = v
	if new_opts['update_param_list']:
		for param in new_opts['update_param_list']:
			ckpt_opts[param] = new_opts[param]


if __name__ == '__main__':
	main()
