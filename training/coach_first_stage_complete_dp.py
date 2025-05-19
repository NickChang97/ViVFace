import os
import pdb
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import common, train_utils
from criteria import id_loss, moco_loss, gradient_variance_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.sample_from_video_dataset import FramesDataset,ImgsDataset,SpecifyOneImgOneDirDataset
from criteria.lpips.lpips import LPIPS
from models.psp_identity_related import pSp
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger
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
        self.net.load_state_dict(ckpt['state_dict'], strict=True)
        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
        print(f'Resuming training from step {self.global_step}')
    def train(self):
        self.net.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                if self.is_training_discriminator():
                    loss_dict = self.train_discriminator(batch)
                S, D1, D2, S_neutral, S_hat, S_D1, D2_S, S_latent, D1_latent, D2_S_latent, D2_latent = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(S, D1, D2, S_neutral, S_hat, S_D1, D2_S, S_latent, D1_latent, D2_S_latent, D2_latent)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    S[1] = S[0]
                    D1[1] = D2[0]
                    S_D1[1] = D2_S[0]
                    self.parse_and_log_images(id_logs, S, D1, S_D1, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    self.checkpoint_me(loss_dict, is_best=False)
                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break
                self.global_step += 1
                if self.opts.progressive_steps:
                    self.check_for_progressive_training_update()
    def inference(self):
        self.net.train()

        save_dir = 'specify_img_inference'
        save_dir = os.path.join(self.opts.exp_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        source_dir = os.path.join(save_dir, 'source')
        os.makedirs(source_dir, exist_ok=True)
        driving_dir = os.path.join(save_dir, 'driving')
        os.makedirs(driving_dir, exist_ok=True)
        driven_dir = os.path.join(save_dir, 'driven')
        os.makedirs(driven_dir, exist_ok=True)
        driving_recon_dir = os.path.join(save_dir, 'driving_recon')
        os.makedirs(driving_recon_dir, exist_ok=True)

        for batch_idx, batch in enumerate(self.test_dataloader):
            print(f"\r {batch_idx} / {len(self.test_dataloader)}", end="")

            S = batch['source']
            D2 = batch['driving']
            image_name = batch['name'][0]
        
            S,  D2 = S.to(self.device).float(), D2.to(self.device).float()
            with torch.no_grad():
                D2_hat, D2_latent = self.net.forward(D2, return_latents=True)
                S_D2 = self.net.forward(S, ss_generic_latent=D2_latent['ss_generic_latent'], return_latents=False, randomize_noise=False)
            
            Source_image = common.tensor2im(S[0])
            Driving_image = common.tensor2im(D2[0])
            Driven_image = common.tensor2im(S_D2[0])
            Driving_recon_image = common.tensor2im(D2_hat[0])

            #image_name = f'{save_idx}.png'

            Source_image_name = os.path.join(source_dir, image_name)
            Driving_image_name = os.path.join(driving_dir, image_name)
            Driving_recon_image_name = os.path.join(driving_recon_dir, image_name)
            Driven_image_name = os.path.join(driven_dir, image_name)

            Source_image.save(Source_image_name)
            Driving_image.save(Driving_image_name)
            Driven_image.save(Driven_image_name)
            Driving_recon_image.save(Driving_recon_image_name)
            
    def image_inverse_1024(self):
        self.net.eval()
        save_dir = 'image_inverse'
        save_dir = os.path.join(self.opts.exp_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        source_dir = os.path.join(save_dir, 'source')
        os.makedirs(source_dir, exist_ok=True)
        driving_dir = os.path.join(save_dir, 'driving')
        os.makedirs(driving_dir, exist_ok=True)
        source_recon_dir = os.path.join(save_dir, 'source_recon')
        os.makedirs(source_recon_dir, exist_ok=True)
        driving_recon_dir = os.path.join(save_dir, 'driving_recon')
        os.makedirs(driving_recon_dir, exist_ok=True)
        for batch_idx, batch in enumerate(self.test_dataloader):
            print(f"\r {batch_idx} / {len(self.test_dataloader)}", end="")
            S = batch['source']
            D2 = batch['driving']
            image_name = batch['name'][0]
            S,  D2 = S.to(self.device).float(), D2.to(self.device).float()
            with torch.no_grad():
                S_hat = self.net.forward(S, return_latents=False, return_images=True, resize=False)
                D2_hat = self.net.forward(D2, return_latents=False, return_images=True, resize=False)
            Source_image = common.tensor2im(S[0])
            Driving_image = common.tensor2im(D2[0])
            Source_recon_image = common.tensor2im(S_hat[0])
            Driving_recon_image = common.tensor2im(D2_hat[0])
            Source_image_name = os.path.join(source_dir, image_name)
            Driving_image_name = os.path.join(driving_dir, image_name)
            Driving_recon_image_name = os.path.join(driving_recon_dir, image_name)
            source_recon_dir_image_name = os.path.join(source_recon_dir, image_name)
            Source_image.save(Source_image_name)
            Driving_image.save(Driving_image_name)
            Source_recon_image.save(source_recon_dir_image_name)
            Driving_recon_image.save(Driving_recon_image_name)
    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  
                self.net.encoder.module.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   
                self.net.encoder.module.set_progressive_stage(ProgressiveStage(i))
    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(batch)
            with torch.no_grad():
                x, y, y_hat, latent = self.forward(batch)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)
            self.parse_and_log_images(id_logs, x, y, y_hat,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')
        self.net.train()
        return loss_dict
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
        params = list(self.net.encoder.parameters())
        params = params + list(self.net.ss_latent_transformer.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer
    def configure_datasets(self):
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
            test_dataset = FramesDataset(dataset_args['root_dir'], frame_shape=dataset_args['frame_shape'], is_train=True,
                 random_seed=0, sample_percent=dataset_args['sample_percent'])
        elif self.opts.dataset_type in [ 'aligned_celevb_image']:
            train_dataset = ImgsDataset(dataset_args['root_dir'], frame_shape=dataset_args['frame_shape'], is_train=True,
                 random_seed=0, sample_percent=dataset_args['sample_percent'])
            test_dataset = ImgsDataset(dataset_args['root_dir'], frame_shape=dataset_args['frame_shape'], is_train=True,
                 random_seed=0, sample_percent=dataset_args['sample_percent'])
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
    def calc_loss(self, S, D1, D2, S_neutral, S_hat, S_D1, D2_S, S_latent, D1_latent, D2_S_latent, D2_latent):
        loss_dict = {}
        id_logs = None
        loss_dict['L_self'] = self.rec_loss(S, S_hat)
        loss_dict['L_reenact'] = self.rec_loss(D1, S_D1)
        loss_dict['L_latent_consistency'] = self.opts.consistency_lambda*self.mse_loss(S_latent['w_latent'], D2_S_latent['w_latent'])
        loss_dict['L_ss_latent_consistency'] = self.opts.consistency_lambda*self.mse_loss(D2_latent['ss_generic_latent'], D2_S_latent['ss_generic_latent'])
        loss_dict['L_ss_latent_regularization'] = self.opts.delta_norm_lambda*self.opts.s_lambda*self.mse_loss(D1_latent['ss_generic_latent'], torch.zeros_like(D1_latent['ss_generic_latent']).cuda()) 
        if self.is_training_discriminator():  
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.net.decoder.n_latent))
            for i in dims_to_discriminate:
                w = S_latent['w_latent'][:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = self.opts.w_discriminator_lambda * loss_disc
        if self.opts.progressive_steps and self.net.encoder.module.progressive_stage.value != 18:  
            total_delta_loss = 0
            deltas_latent_dims = self.net.encoder.module.get_deltas_starting_dimensions()
            first_w = S_latent['w_latent'][:, 0, :]
            for i in range(1, self.net.encoder.module.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = S_latent['w_latent'][:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = self.opts.delta_norm_lambda * delta_loss
        if self.opts.id_lambda > 0:  
            loss_id_a, sim_improvement, id_logs = self.id_loss(S_neutral, S, S_neutral)
            loss_id_b, sim_improvement, _ = self.id_loss(D2_S, S, D2_S)
            loss_dict['loss_id'] = (loss_id_a + loss_id_b) * self.opts.id_lambda
        loss_dict['loss'] = torch.tensor(0.0).cuda()
        for each_key in loss_dict:
            if each_key != 'loss':
                loss_dict['loss'] += loss_dict[each_key]
        return loss_dict['loss'], loss_dict, id_logs
    def forward(self, batch):
        S = batch['source']
        D1 = batch['driving']
        D2 = batch['other_identity']
        S, D1, D2 = S.to(self.device).float(), D1.to(self.device).float(), D2.to(self.device).float()
        S_hat, S_latent = self.net.forward(S, return_latents=True)
        D_hat, D1_latent = self.net.forward(D1, return_latents=True)
        S_D1 = self.net.forward(S, skip_latent=True, w_latent=S_latent['w_latent'],input_memory=S_latent['input_memory'],ss_generic_latent=D1_latent['ss_generic_latent'], return_latents=False)
        zero_ss_latent = torch.zeros_like(S_latent['ss_latent']).cuda()
        if self.opts.id_lambda != 0.0:
            S_neutral = self.net.forward(S, skip_latent=True, \
                                            w_latent=S_latent['w_latent'],\
                                            ss_generic_latent=zero_ss_latent, \
                                            input_memory=S_latent['input_memory'],
                                            return_latents=False)
            D2_S, D2_latent = self.net.forward(D2, w_latent=S_latent['w_latent'], return_latents=True, input_memory=S_latent['input_memory']) 
        else:
            S_neutral = None
            D2_S = None
        D2_S_latent = self.net.forward(D2_S, return_latents=True, return_images=False) 
        return S, D1, D2, S_neutral, S_hat, S_D1, D2_S, S_latent, D1_latent, D2_S_latent, D2_latent
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
        fake_w, _, _ = self.net.encoder(x)
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
