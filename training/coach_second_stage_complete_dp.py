import os
import sys
import pdb
import random
import matplotlib
import matplotlib.pyplot as plt
import tqdm
matplotlib.use('Agg')
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from argparse import Namespace
from utils import common, train_utils
from criteria import id_loss, moco_loss, gradient_variance_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.sample_from_video_dataset import FramesDataset,ImgsDataset,SpecifyOneImgOneDirDataset
from criteria.lpips.lpips import LPIPS
from models.psp_identity_related_HFGI_v5_fix_ss_style import pSp
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger
import clip
from hairclip.mapper.hairclip_mapper import HairCLIPMapper
random.seed(0)
torch.manual_seed(0)

class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts
        self.global_step = 0
        self.device = 'cuda:0'
        self.opts.device = self.device
        self.net = pSp(self.opts).to(self.device)
        self.net.encoder = torch.nn.DataParallel(self.net.encoder)
        self.net.decoder = torch.nn.DataParallel(self.net.decoder)
        hairclip_ckpt = torch.load(self.opts.hairclip_checkpoint_path, map_location='cpu')
        hairclip_opts = hairclip_ckpt['opts']
        hairclip_opts['checkpoint_path'] = self.opts.hairclip_checkpoint_path
        hairclip_opts['editing_type'] = 'hairstyle'
        hairclip_opts['input_type'] = 'text'
        hairclip_opts['hairstyle_description'] = 'hairstyle_list.txt'
        hairclip_opts['no_coarse_mapper'] = False
        hairclip_opts['no_medium_mapper'] = False
        hairclip_opts['no_fine_mapper'] = False
        hairclip_opts = Namespace(**hairclip_opts)
        self.hairclip_net = HairCLIPMapper(hairclip_opts)
        self.hairclip_net.eval()
        self.hairclip_net.cuda()
        self.color_text_inputs = torch.tensor([[0.0]]).float().cuda()
        self.hairstyle_tensor_hairmasked = torch.tensor([[0.0]]).float().cuda()
        self.color_tensor_hairmasked = torch.tensor([[0.0]]).float().cuda()
        self.hairstyle_text_file = 'mapper/hairstyle_list.txt'
        fr = open(self.hairstyle_text_file, 'r')
        self.hairstyle_texts = [each_line.strip() for each_line in fr]
        fr.close()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        self.mse_loss = nn.MSELoss().to(self.device)
        self.grad_criterion = gradient_variance_loss.GradientVariance(patch_size=self.opts.loss_patch_size) 
        self.optimizer = self.configure_optimizers()
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps
        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None
    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.net.load_state_dict(ckpt['state_dict'])
        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
    def train(self):
        self.net.train()
        w_age = torch.load('/root/data/.shared_8t/qing_chang/HFGI-main/editings/interfacegan_directions/age.pt','cuda')
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                S, S_hair_aug, S_hat, ori_recon_images, rec, D = self.forward(batch, w_age=w_age)
                loss, encoder_loss_dict, id_logs = self.calc_loss(S, S_hat, rec)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 100 == 0):
                    ori_recon_images = torch.nn.functional.interpolate(torch.clamp(ori_recon_images, -1., 1.), size=(256,256) , mode='bilinear') 
                    batch['source'][1] = S_hair_aug[0]
                    S[1] = ori_recon_images[0] 
                    S_hat[1] = D[0]
                    self.parse_and_log_images(id_logs, batch['source'][:2], S, S_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    self.checkpoint_me(loss_dict, is_best=False)
                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break
                self.global_step += 1
    def image_inverse_1024(self):
        self.net.eval()
        save_dir = os.path.join(self.opts.exp_dir, 'image_inverse')
        os.makedirs(save_dir, exist_ok=True)
        for batch_idx, batch in tqdm.tqdm(enumerate(self.test_dataloader)):
            loss_dict = {}
            batch['source'] = batch['driving']
            batch['other_identity'] = batch['driving']
            with torch.no_grad():
                S, S_hair_aug, S_hat, ori_recon_images = self.forward(batch, resize=False)
            S_img = common.tensor2im(S[0])
            S_hat_img = common.tensor2im(S_hat[0])
            ori_recon_images = common.tensor2im(ori_recon_images[0])
            S_img.save(f'{save_dir}/{batch_idx}_S.png')
            S_hat_img.save(f'{save_dir}/{batch_idx}_S_hat_img.png')
            ori_recon_images.save(f'{save_dir}/{batch_idx}_ori_recon_images.png')
    def other_driven_1024_makeup(self):
        self.net.eval() 
        save_dir = os.path.join(self.opts.exp_dir, 'other_driven-use-resize_random-noise-false')
        os.makedirs(save_dir, exist_ok=True)
        global_idx = 0
        for batch_idx, batch in tqdm.tqdm(enumerate(self.test_dataloader)):
            S = batch['source']
            D1 = batch['driving']
            if self.opts.edit_direction == 'age':
                w_edit = torch.load('/root/data2/jiajun_sun/HFGI-main/editings/interfacegan_directions/age.pt','cuda')
                w_edit = self.opts.edit_extent * w_edit
                w_edit = w_edit.unsqueeze(1).repeat(S.shape[0],1,1)
            S, D1 = S.to(self.device).float(), D1.to(self.device).float()
            with torch.no_grad():
                D1_latent = self.net.image_inverse(D1, return_latents=True, return_images=False)
                S_latent = self.net.image_inverse(S, return_latents=True, return_images=False, resize=False)
                S_D1 = self.net.image_inverse(S, ss_latent=D1_latent['ss_latent'], resize=False, return_latents=False, return_images=True)
                refine_S_D1 = self.net.refine_driven(S_D1, S, S_latent['w_latent']+w_edit, D1_latent['ss_latent'], resize=False)
            for sample_idx in range(S.shape[0]):
                S_img = common.tensor2im(S[sample_idx])
                D1_img = common.tensor2im(D1[sample_idx])
                S_D1_img = common.tensor2im(S_D1[sample_idx])
                refine_S_D1_img = common.tensor2im(refine_S_D1[sample_idx])
                S_img.save(f'{save_dir}/{global_idx}_S.png')
                D1_img.save(f'{save_dir}/{global_idx}_D.png')
                S_D1_img.save(f'{save_dir}/{global_idx}_S_D1.png')
                refine_S_D1_img.save(f'{save_dir}/{global_idx}_S_D1_refine.png')
                global_idx += 1
    def other_driven_1024_use_resize(self):
        self.net.eval() 
        save_dir = os.path.join(self.opts.exp_dir, 'other_driven-use-resize_random-noise-false')
        os.makedirs(save_dir, exist_ok=True)
        global_idx = 0
        for batch_idx, batch in tqdm.tqdm(enumerate(self.test_dataloader)):
            S = batch['source']
            D1 = batch['driving']
            S, D1 = S.to(self.device).float(), D1.to(self.device).float()
            with torch.no_grad():
                D1_latent = self.net.image_inverse(D1, return_latents=True, return_images=False)
                S_D1, S_latent = self.net.image_inverse(S, ss_generic_latent=D1_latent['ss_generic_latent'], resize=False, return_latents=True, return_images=True)
                refine_S_D1 = self.net.refine_driven(S_D1, S, S_latent['w_latent'], ss_codes=S_latent['ss_latent'], resize=False)
            for sample_idx in range(S.shape[0]):
                S_img = common.tensor2im(S[sample_idx])
                D1_img = common.tensor2im(D1[sample_idx])
                S_D1_img = common.tensor2im(S_D1[sample_idx])
                refine_S_D1_img = common.tensor2im(refine_S_D1[sample_idx])
                S_img.save(f'{save_dir}/{global_idx}_S.png')
                D1_img.save(f'{save_dir}/{global_idx}_D.png')
                S_D1_img.save(f'{save_dir}/{global_idx}_S_D1.png')
                refine_S_D1_img.save(f'{save_dir}/{global_idx}_S_D1_refine.png')
                global_idx += 1
    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  
                self.net.encoder.module.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   
                self.net.encoder.module.set_progressive_stage(ProgressiveStage(i))
    def validate(self):
        pass
    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))
    def configure_optimizers(self):
        params = list(self.net.residue.parameters())
        params  += list(self.net.grid_align.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer
    def configure_datasets(self, hairclip_net=None):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        if self.opts.driving_image_dir != '':
            dataset_args['driving_dir'] = self.opts.driving_image_dir
        if self.opts.source_image != '':
            dataset_args['source_image'] = self.opts.source_image
        if self.opts.dataset_type in ['celevb_video', 'aligned_celevb_video']:
            train_dataset = FramesDataset(dataset_args['root_dir'], frame_shape=dataset_args['frame_shape'], is_train=True,
                 random_seed=0, sample_percent=dataset_args['sample_percent'])
            test_dataset = FramesDataset(dataset_args['root_dir'], frame_shape=dataset_args['frame_shape'], is_train=False,
                 random_seed=0, sample_percent=dataset_args['sample_percent'])
        elif self.opts.dataset_type in [ 'aligned_celevb_image']:
            train_dataset = ImgsDatasetV2(dataset_args['root_dir'], frame_shape=dataset_args['frame_shape'], is_train=True,
                 random_seed=0, sample_percent=dataset_args['sample_percent'], load_mask=False, load_e4e_latent=True)
            test_dataset = ImgsDatasetV2(dataset_args['root_dir'], frame_shape=dataset_args['frame_shape'], is_train=False,
                 random_seed=0, sample_percent=dataset_args['sample_percent'], load_mask=False)
        elif self.opts.dataset_type in [ 'specify_source_image_driving_dir']:
            train_dataset = SpecifyOneImgOneDirDataset(dataset_args['source_image'],dataset_args['driving_dir'])
            test_dataset = SpecifyOneImgOneDirDataset(dataset_args['source_image'],dataset_args['driving_dir'])
        else:
            transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
            train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                          target_root=dataset_args['train_target_root'],
                                          source_transform=transforms_dict['transform_source'],
                                          target_transform=transforms_dict['transform_gt_train'],
                                          opts=self.opts)
            test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                         target_root=dataset_args['test_target_root'],
                                         source_transform=transforms_dict['transform_source'],
                                         target_transform=transforms_dict['transform_test'],
                                         opts=self.opts)
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset
    def rec_loss(self, source, target):
        return self.opts.l2_lambda*self.mse_loss(source, target) \
               + self.opts.lpips_lambda*self.lpips_loss(source, target) \
               + self.opts.gv_lambda*self.grad_criterion(source, target)
    def calc_loss(self, S, S_hat, res_delta):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:  
            loss_id, sim_improvement, id_logs = self.id_loss(S_hat, S, S)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(S_hat, S)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(S_hat, S)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.res_lambda > 0:
            target = torch.zeros_like(res_delta)
            loss_res = F.l1_loss(res_delta, target)
            loss_dict['loss_res'] = float(loss_res)
            loss += loss_res * self.opts.res_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
    def forward(self, batch, w_age=None, resize=True):
        S = batch['source']
        D1 = batch['driving']
        D2 = batch['other_identity']
        S, D1, D2 = S.to(self.device).float(), D1.to(self.device).float(), D2.to(self.device).float()
        aug_prob = random.random()
        if aug_prob < self.opts.hair_aug_rate:
            with torch.no_grad():
                S_w = batch['source_e4e_latent'].cuda().float().squeeze(1)
                hairstyle_text_inputs = clip.tokenize([random.choice(self.hairstyle_texts)]).cuda()
                hairstyle_text_inputs = hairstyle_text_inputs.repeat(S.shape[0], 1)
                S_hat, S_latent = self.net.image_inverse(S, return_latents=True, return_images=True, resize=False)
                w_edit = 0.1 * self.hairclip_net.mapper(S_w, hairstyle_text_inputs, self.color_text_inputs, self.hairstyle_tensor_hairmasked, self.color_tensor_hairmasked)
                S_hair_aug = self.net.image_inverse(S, w_latent=S_latent['w_latent']+w_edit, resize=True, return_latents=False, return_images=True)
            S_hat, ori_recon_images, rec = self.net.forward(S, S_hair_aug, D1, \
                                                            w_latent=S_latent['w_latent']+w_edit,\
                                                            ss_latent=S_latent['ss_latent'],\
                                                            return_latents=True, \
                                                            resize=resize, \
                                                            skip_latent=True)
            return S, S_hair_aug, S_hat, ori_recon_images, rec, D1
        elif aug_prob < self.opts.age_aug_rate:
            with torch.no_grad():
                S_hat, S_latent = self.net.image_inverse(S, return_latents=True, return_images=True, resize=False)
                w_edit = (1.0+random.random()*1.5) * w_age 
                w_edit = random.choice([-1,1]) * w_edit
                w_edit = w_edit.unsqueeze(1).repeat(S.shape[0],1,1)
                S_age_aug = self.net.image_inverse(S, w_latent=S_latent['w_latent']+w_edit, resize=True, return_latents=False, return_images=True)
            S_hat, ori_recon_images, rec = self.net.forward(S, S_age_aug, D1, \
                                                            w_latent=S_latent['w_latent']+w_edit,\
                                                            ss_latent=S_latent['ss_latent'],\
                                                            return_latents=True, \
                                                            resize=resize, \
                                                            skip_latent=True)
            return S, S_age_aug, S_hat, ori_recon_images, rec, D1
        else:
            with torch.no_grad():
                S_hat, S_latent = self.net.image_inverse(S, return_latents=True, return_images=True, resize=False)
                w_edit = torch.nn.functional.normalize(torch.randn(1,512)+1.0, dim=1).cuda()
                w_edit = self.opts.edit_extent * random.random() * w_edit 
                w_edit = w_edit.unsqueeze(1).repeat(S.shape[0],1,1)
                S_age_aug = self.net.image_inverse(S, w_latent=S_latent['w_latent']+w_edit, resize=True, return_latents=False, return_images=True)
            S_hat, ori_recon_images, rec = self.net.forward(S, S_age_aug, D1, \
                                                            w_latent=S_latent['w_latent']+w_edit,\
                                                            ss_latent=S_latent['ss_latent'],\
                                                            return_latents=True, \
                                                            resize=resize, \
                                                            skip_latent=True)
            return S, S_age_aug, S_hat, ori_recon_images, rec, D1
    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)
    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)
    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)
    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        if self.opts.save_training_data:  
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return save_dict
    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.net.encoder.module.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.net.encoder.module.progressive_stage.value + 1]
    def is_progressive_training(self):
        return self.opts.progressive_steps is not None
    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0
    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()
        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)
        return real_loss + fake_loss
    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty
    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
    def train_discriminator(self, batch):
        loss_dict = {}
        x = batch['source']
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)
        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)
            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)
        self.requires_grad(self.discriminator, False)
        return loss_dict
    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _ = test_batch
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict
    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        real_w = self.net.decoder.module.get_latent(sample_z)
        fake_w, _ = self.net.encoder(x)
        if self.opts.start_from_latent_avg:
            fake_w = fake_w + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
        if self.is_progressive_training():  
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w
