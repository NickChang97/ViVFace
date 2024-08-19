from argparse import ArgumentParser
from configs.paths_config import model_paths
class TrainOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()
    def initialize(self):
        self.parser.add_argument('--source_image', type=str, help='Path to source image')
        self.parser.add_argument('--driving_image_dir', type=str, help='Path to dir of driving image')
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='Encoder4Editing', type=str, help='Which encoder to use')
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--res_lambda', default=0., type=float, help='L2 loss multiplier factor')  
        self.parser.add_argument('--edit_extent', default=0.1, type=float, help='')
        self.parser.add_argument('--edit_direction', default='bowl cut hairstyle', type=str, help='')
        self.parser.add_argument('--distortion_scale', type=float, default=0.15, help="lambda for delta norm loss")
        self.parser.add_argument('--ffhq_distortion_scale', type=float, default=0.15, help="lambda for delta norm loss")
        self.parser.add_argument('--aug_rate', type=float, default=0.8, help="lambda for delta norm loss")
        self.parser.add_argument('--hair_aug_rate', type=float, default=0.5, help="lambda for delta norm loss")
        self.parser.add_argument('--age_aug_rate', type=float, default=0.5, help="lambda for delta norm loss")
        self.parser.add_argument('--ffhq_aug_rate', type=float, default=0.0, help="lambda for delta norm loss")
        self.parser.add_argument('--slim_aug', type=float, default=0.5, help="probability of slim agumentation when choosing slim or fatten")
        self.parser.add_argument('--stylegan_supplemented_index', type=int, default=7, help='.')
        self.parser.add_argument('--residue_output_dim', type=int, default=512, help='')
        self.parser.add_argument('--residue_output_size', type=int, default=64, help='')
        self.parser.add_argument('--hr_stylegan_supplemented_index', type=int, default=13, help='.')
        self.parser.add_argument('--hr_residue_output_dim', type=int, default=64, help='')
        self.parser.add_argument('--hr_residue_output_size', type=int, default=512, help='')
        self.parser.add_argument('--previous_supplemented_frozen', action='store_true',
                                 help='Whether to store a latnet codes pool for the discriminator\'s training')
        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--stylegan_size', default=1024, type=int,
                                 help='size of pretrained StyleGAN Generator')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
        self.parser.add_argument('--hairclip_checkpoint_path', default=None, type=str, help='Path to HairClip model checkpoint')
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')
        self.parser.add_argument('--ss_latent_contrastive_lambda', type=float, default=0.5)
        self.parser.add_argument('--import_region_lambda', type=float, default=0.5)
        self.parser.add_argument('--gv_lambda', default=0.1, type=float, help='gv multiplier factor')
        self.parser.add_argument('--ss_styles', type=int, default=10, help='.')
        self.parser.add_argument('--w_consistency_lambda', type=float, default=0.2)
        self.parser.add_argument('--s_lambda', type=float, default=0.2)
        self.parser.add_argument('--loss_patch_size', type=int, default=8, help='.')
        self.parser.add_argument('--consistency_lambda', type=float, default=1.0)
        self.parser.add_argument('--ss_generic_consistency_lambda', type=float, default=1.0)
        self.parser.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
        self.parser.add_argument('--w_discriminator_lr', default=2e-5, type=float, help='Dw learning rate')
        self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        self.parser.add_argument("--d_reg_every", type=int, default=16,
                                 help="interval for applying r1 regularization")
        self.parser.add_argument('--use_w_pool', action='store_true',
                                 help='Whether to store a latnet codes pool for the discriminator\'s training')
        self.parser.add_argument("--w_pool_size", type=int, default=50,
                                 help="W\'s pool size, depends on --use_w_pool")
        self.parser.add_argument('--delta_norm', type=int, default=2, help="norm type of the deltas")
        self.parser.add_argument('--delta_norm_lambda', type=float, default=2e-4, help="lambda for delta norm loss")
        self.parser.add_argument('--progressive_steps', nargs='+', type=int, default=None,
                                 help="The training steps of training new deltas. steps[i] starts the delta_i training")
        self.parser.add_argument('--progressive_start', type=int, default=None,
                                 help="The training step to start training the deltas, overrides progressive_steps")
        self.parser.add_argument('--progressive_step_every', type=int, default=2_000,
                                 help="Amount of training steps for each progressive step")
        self.parser.add_argument('--save_training_data', action='store_true',
                                 help='Save intermediate training data to resume training from the checkpoint')
        self.parser.add_argument('--sub_exp_dir', default=None, type=str, help='Name of sub experiment directory')
        self.parser.add_argument('--keep_optimizer', action='store_true',
                                 help='Whether to continue from the checkpoint\'s optimizer')
        self.parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                 help='Path to training checkpoint, works when --save_training_data was set to True')
        self.parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                 help="Name of training parameters to update the loaded training checkpoint")
    def parse(self):
        opts = self.parser.parse_args()
        return opts
