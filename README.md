# ViVFace
Codes for Enhancing Identity-Deformation Disentanglement in StyleGAN for One-Shot Face Video Re-Enactment. 

Project Page https://nickchang97.github.io/ViVFace.github.io/

ViVFace is accepted by AAAI 2025


## TODO
- [x] Data preparation
- [x] Training codes
- [x] Inference codes and models
- [ ] Examples

## DATASET

Images should be organized as this:
HDTF

	--identity1
 
		xxx.png
  
	--identity2
 
		xxx.png
  
	...
 
CelebVHQ

	--identity1
 
		xxx.png
  
	--identity2
 
		xxx.png
  
	...

specify data root in configs/paths_config.py

## Training
Stage-I
```.bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --dataset_type aligned_celevb_image --exp_dir second_stage  --start_from_latent_avg --use_w_pool --w_discriminator_lambda 0.1 --gv_lambda 1e-2 --consistency_lambda 1.0 --delta_norm_lambda 2e-4 --s_lambda 1.0 --import_region_lambda 0.0 --id_lambda 0.1 --val_interval 100000000 --max_steps 100000 --stylegan_size 1024 --workers 16 --batch_size 8 --test_batch_size 4 --test_workers 4 --learning_rate 0.0001 --save_training_data --save_interval 10000 --ss_latent_contrastive_lambda 0.00 --w_discriminator_lr 2e-5  --image_interval 100
```

Stage-II
```.bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_second_stage.py --dataset_type aligned_celevb_image --exp_dir second_stage  --start_from_latent_avg --use_w_pool --val_interval 100000000 --max_steps 100000 --stylegan_size 1024 --checkpoint_path first_stage.pt --workers 16 --batch_size 8 --test_batch_size 4 --test_workers 4 --learning_rate 0.0001 --save_training_data --save_interval 10000 --ss_latent_contrastive_lambda 0.00 --w_discriminator_lr 2e-5 --aug_rate 0.9 --res_lambda 0.1 --hairclip_checkpoint_path pretrained_models/hairclip.pt --hair_aug_rate 0.33 --age_aug_rate 0.67 --image_interval 100
```

## Inference
```.bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_ours.py --dataset_type=specify_source_image_driving_dir --source_image xxx.png --driving_image_dir xxx --exp_dir=xxx --workers=0 --batch_size=1 --test_batch_size=1 --test_workers=0 --val_interval=2500 --save_interval=5000 --resume_training_from_ckpt CKPT_PATH
```

Please note that, images should take face-alignment according to FFHQ dataset due to the characteristics when training StyleGAN.

checkpoint link: https://pan.baidu.com/s/1kGOs9m_2R2pKEdX9_Gl57g Code: xwsg 

## Citation

If you find this project useful in your research, please consider cite:

```latex
@inproceedings{chang2025enhancing,
  title={Enhancing Identity-Deformation Disentanglement in StyleGAN for One-Shot Face Video Re-Enactment},
  author={Chang, Qing and Ding, Yao-Xiang and Zhou, Kun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={1247--1255},
  year={2025}
}
```
