from config.dfvq4imagenet import *
from einops import rearrange
import numpy as np
from collections import namedtuple
from torchvision import models, transforms
from torch.nn import init
import einops
from config.dfvqgan import MaskEncoder, MaskDecoder, LPIPS, NLayerDiscriminator, VectorQuantizer, Encoder

@iterable_class
class Args(BasicArgs):
    task_name, method_name, log_dir = BasicArgs.get_log_dir(__file__)
    pretrained_encoder = False
    fix_encoder = False

    load_mem = False

    max_train_samples = None
    max_eval_samples = 32

    train_batch_size = 48
    eval_batch_size = 16
    learning_rate = 4.5e-06
    gradient_accumulate_steps = 1
    epochs = 5000
    seed = 42
    num_workers = 1
    eval_step = 1
    save_step = 1

    epoch_threshold = 2
    codebook_weight = 1.0
    discriminator_weight = 0.8
    perceptual_weight = 1.0

    adapt_load = False

    img_size = 256  # 256/16=16
    coder_config = {'double_z': False, 'z_channels': 256, 'resolution': img_size, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                    'ch_mult': [1, 1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16], 'dropout': 0.0}

    quantize_config = {'n_e': 8192, 'e_dim': 256, 'beta': 0.25, 'remap': None, 'sane_index_shape': False, 'legacy': False}
    discriminator_config = {'input_nc': 3, 'n_layers': 3, 'use_actnorm': False, 'ndf': 64}
    find_unused_parameters = True
    patch_size = 8

    min_vmlm_ratio = 0.1
    max_vmlm_ratio = 0.4

    full_mask_ratio = 0.2
    random_mask_ratio = 0.4

    none_decoder_ratio = 0.5

    set_seed(seed)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.05, contrast=0.25, saturation=0.25, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

args = Args()

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake)))
    return d_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.encoder = MaskEncoder(**args.coder_config)
        self.decoder = MaskDecoder(**args.coder_config)
        self.epoch_threshold = args.epoch_threshold
        self.codebook_weight = args.codebook_weight
        self.discriminator_weight = args.discriminator_weight
        self.perceptual_weight = args.perceptual_weight

        self.perceptual_loss = LPIPS().eval()
        self.discriminator = NLayerDiscriminator(**args.discriminator_config).apply(weights_init)

        self.quantize = VectorQuantizer(**args.quantize_config)
        self.quant_conv = torch.nn.Conv2d(args.coder_config["z_channels"], args.quantize_config["e_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(args.quantize_config["e_dim"], args.coder_config["z_channels"], 1)

        if self.args.fix_encoder:
            for n, p in self.encoder.named_parameters():
                p.requires_grad = False
            for n, p in self.quantize.named_parameters():
                p.requires_grad = False

        real_learning_rate = args.gradient_accumulate_steps * args.train_batch_size * args.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=real_learning_rate, betas=(0.5, 0.9))
        opt_ae.is_enabled = lambda epoch: True
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=real_learning_rate, betas=(0.5, 0.9))
        opt_disc.is_enabled = lambda epoch: epoch >= self.epoch_threshold
        self.optimizers = [opt_ae, opt_disc]
        # self.optimizers = [opt_ae]
        self.best_metric_fn = lambda x: x['train']['loss_total']
        self.scheduler = None

    def load_state_dict(self, state):
        try:
            super(Net, self).load_state_dict(state)
        except RuntimeError as e:
            if self.args.pretrained_encoder:
                new_state = OrderedDict()
                for key in state:
                    if "decoder" not in key:
                        new_state[key] = state[key]
                super(Net, self).load_state_dict(new_state, strict=False)
            else:
                raise e

    @torch.no_grad()
    def get_codebook_indices(self, img, mask):
        mask = mask.bool()
        b = img.shape[0]
        # img = (2 * img) - 1
        h = self.encoder(img, mask)
        h = self.quant_conv(h)
        _, _, [_, _, indices] = self.quantize(h)
        return rearrange(indices.flatten(), '(b n)-> b n', b=b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes=self.args.quantize_config['n_e']).float()
        z = (one_hot_indices @ self.quantize.embedding.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h=int(math.sqrt(n)))
        quant = self.post_quant_conv(z)
        img = self.decoder(quant)
        # img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, inputs):
        x = inputs['input_img'].to(memory_format=torch.contiguous_format)
        mask = inputs['vision_mask'].bool()
        device = x.device
        # xrec, qloss = self(x)
        if inputs["split"][0] != "train":
            mask = torch.ones_like(mask).bool()
        h, hs, hms = self.encoder(x, mask)
        h = self.quant_conv(h)
        quant_full, qloss, _ = self.quantize(h)
        quant = self.post_quant_conv(quant_full)

        for idx, dt in enumerate(inputs["decoder_type"]):
            if dt == "none":
                for idx2 in range(len(hms)):
                    tmp = hms[idx2].clone()
                    tmp[idx] = True#torch.ones_like(tmp[idx2]).to(device).bool()
                    hms[idx2] = tmp

        xrec = self.decoder(quant, hs, hms)

        if self.training and inputs['epoch'] < self.epoch_threshold:
            disc_factor = 0.0
        else:
            disc_factor = 1.0
        outputs = {}
        outputs['logits_img'] = xrec

        if not "optimizer_idx" in inputs or inputs['optimizer_idx'] == 0:
            # Reconstruction Loss
            rec_loss = torch.abs(x.contiguous() - xrec.contiguous()).mean()
            # Perceptual Loss
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous()).mean()
            else:
                p_loss = torch.tensor([0.0], device=device)
            # NLL Loss
            nll_loss = rec_loss + self.perceptual_weight * p_loss
            last_layer = self.decoder.conv_out.weight

            if disc_factor:
                logits_fake = self.discriminator(xrec.contiguous())
                g_loss = -torch.mean(logits_fake)
                if self.training:
                    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
                    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
                    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
                    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
                    d_weight = d_weight * self.discriminator_weight
                else:
                    d_weight = torch.tensor(0.0, device=device)
                loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * qloss.mean()
            else:
                g_loss = torch.tensor(0.0, device=device)
                loss = nll_loss + self.codebook_weight * qloss.mean()

            outputs['loss_total'] = loss
            outputs['loss_q'] = qloss
            outputs['loss_rec'] = rec_loss
            outputs['loss_p'] = p_loss
            outputs['loss_nll'] = nll_loss
            outputs['loss_g'] = g_loss

        if not "optimizer_idx" in inputs or inputs['optimizer_idx'] == 1:
            # Reconstruction Loss
            # Perceptual Loss
            # TODO Removed next two lines since they are useless
            # if self.perceptual_weight > 0:
            #     p_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous()).mean()
            # else:
            #     p_loss = torch.tensor([0.0], device=device)

            # discriminator
            # TODO Removed detch() for logits_fake
            if disc_factor:
                logits_real = self.discriminator(x.contiguous().detach())
                logits_fake = self.discriminator(xrec.contiguous())
                d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)
                # d_loss = disc_factor * vanilla_d_loss(logits_real, logits_fake)
            else:
                d_loss = torch.tensor(0.0, device=device)
            outputs['loss_total_1'] = d_loss
            outputs['loss_disc'] = d_loss

        if inputs["split"][0] != "train":
            # vis masked pic
            x = inputs['masked_image'].to(memory_format=torch.contiguous_format)
            mask = inputs['vision_mask'].bool()
            h, hs, hms = self.encoder(x, mask)
            h = self.quant_conv(h)
            quant_mask, qloss, _ = self.quantize(h)
            vae_mask = repeat(inputs['vae_mask'], "b w h -> b c w h", c=self.args.quantize_config["e_dim"])
            quant_all = quant_mask * ~vae_mask.bool() + quant_full * vae_mask.bool()
            quant_all = self.post_quant_conv(quant_all)
            xrec_moreinfo = self.decoder(quant_all, hs, hms)
            xrec_noinfo = self.decoder(quant_all, None, None)
            outputs['logits_imgmaskmoreinfo'] = xrec_moreinfo
            outputs['logits_imgmasknoinfo'] = xrec_noinfo

        return outputs