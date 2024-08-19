import pdb
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import psp_encoders_identity_related
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    key_list = list(d_filt.keys())
    for each_key in key_list:
        if each_key.startswith('module.'):
            d_filt[each_key[7:]] = d_filt.pop(each_key)
    return d_filt
class pSp(nn.Module):
    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        self.encoder = self.set_encoder()
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.ss_latent_transformer = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=4)
        self.load_weights()
    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders_identity_related_fix_ss_style.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders_identity_related_fix_ss_style.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders_identity_related_fix_ss_style.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder
    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            try:
                self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            except Exception as result:
                print(result, "check if this meet expectation")
                self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=False)
            try:
                self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            except Exception as result:
                print(result, 'please check, whether this meet expection')
                self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=False)
            try:
                self.ss_latent_transformer.load_state_dict(get_keys(ckpt, 'ss_latent_transformer'), strict=True)
            except:
                print("check if ss_latent_transformer need be loaded")
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)
    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, skip_latent=False,
                return_images=True, w_latent=None, ss_generic_latent=None, 
                ss_latent=None, input_memory=None):
        if not skip_latent:
            w_codes, ss_generic_codes, ref_feature = self.encoder(x)  
            ref_feature = ref_feature.reshape(ref_feature.shape[0],ref_feature.shape[1],-1)
            ref_feature = ref_feature.permute(2,0,1)
            if self.opts.start_from_latent_avg:
                if w_codes.ndim == 2:
                    w_codes = w_codes + self.latent_avg.repeat(w_codes.shape[0], 1, 1)[:, 0, :]
                else:
                    w_codes = w_codes + self.latent_avg.repeat(w_codes.shape[0], 1, 1)
            returned_latent = {'w_latent':w_codes, 'ss_generic_latent':ss_generic_codes,\
                              'input_memory':ref_feature}
        if w_latent is not None:
            w_codes = w_latent
        if input_memory is not None:
            ref_feature = input_memory
        if ss_generic_latent is not None:
            ss_generic_codes = ss_generic_latent
        if ss_latent is not None:
            assert ss_generic_latent is None
            ss_codes = ss_latent
        else:
            ss_codes = self.ss_latent_transformer(ss_generic_codes.permute(1,0,2), ref_feature)
            ss_codes = ss_codes.permute(1,0,2)
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        w_codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * w_codes[:, i]
                    else:
                        w_codes[:, i] = inject_latent[:, i]
                else:
                    w_codes[:, i] = 0
        input_is_latent = not input_code
        if return_images:
            images, _ = self.decoder([w_codes, ss_codes],
                                                 input_is_latent=input_is_latent, 
                                                 randomize_noise=randomize_noise, 
                                                 return_latents=return_latents)  
            if resize:
                images = self.face_pool(images)
        if return_images and  return_latents:
            returned_latent['ss_latent'] = ss_codes
            return images, returned_latent
        elif return_latents:
            returned_latent['ss_latent'] = ss_codes
            return returned_latent
        elif return_images:
            return images
    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            with torch.no_grad():
                self.latent_avg = self.decoder.mean_latent(10000).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
