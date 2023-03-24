## Forked from oldvq_mar

import sys
from numpy.core.numeric import False_
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

sys.path.append('../..')
from config import *
from config.dfvqgan import DFVQGAN8192
from config.mps4coco import *

from inspect import isfunction

from functools import partial
from itertools import islice, cycle

from math import log2, sqrt, ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F
import clip

from einops import rearrange, repeat
from axial_positional_embedding import AxialPositionalEmbedding
# from dalle_pytorch.transformer import Transformer
from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence
from dalle_pytorch.attention import Attention, SparseAttention, SparseConvCausalAttention

import megatron
from megatron import mpu



@iterable_class
class Args(BasicArgs):
    pretrained_model = os.path.join(BasicArgs.root_dir, "checkpoint/pretrained/epoch66-chunk-0.pth")
    task_name, method_name, log_dir = BasicArgs.get_log_dir(__file__)
    max_train_samples = None
    max_eval_samples = 32
    max_source_len = 35
    img_size = 256
    vqvae_vocab_size = 8192

    load_nuwa = False
    adapt_load = False

    dim = 1280
    dec_depth = 24
    enc_depth = 12
    heads = 20
    dim_head = 64
    attn_dropout = 0.1
    ff_dropout = 0.1
    ignore_index = -100

    attn_types_dec = ('full', 'nearby', 'nearby', 'nearby')
    attn_types_enc = ('full')
    kernel_size = 11

    train_batch_size = 8
    eval_batch_size = 8
    learning_rate = 5e-4
    epochs = 5000
    seed = 42
    num_workers = 1
    eval_step = 10
    save_step = 10

    tk = 128  # How many top logits we consider to sample.
    sample_K = 2  # How many times we sample
    best_n = 2  # How many times we visu for sample_K.
    temperature = 1
    model_path = os.path.join(BasicArgs.root_dir, "CLIP")

    patch_size = 8
    vae_patch_size = 8
    vit_patch_size = 16

    object_mask_ratio = 0

    min_vmlm_ratio = 0.5
    max_vmlm_ratio = 0.7
    min_per_mask_ratio = 0.1
    max_per_mask_ratio = 0.3
    attempt = 5

    # set_seed(seed)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.65, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.25, saturation=0.25, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_prompt = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.65, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
    ])


args = Args()


# helpers


def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cast_tuple(val, depth=1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth

def top_k(logits, tk=1):
    # num_logits = logits.shape[-1]
    # k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, tk)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def generate_nearby_mask(patch_num_row=16, kernel_size=5):
    # nearby mask generation
    effective_kernel_size = kernel_size
    padding = effective_kernel_size // 2
    mask_block = torch.ones(patch_num_row * patch_num_row, patch_num_row + padding * 2,
                            patch_num_row + padding * 2).bool()  # [2560,10,16+padding*2,16+padding*2]
    for i in range(patch_num_row):
        for j in range(patch_num_row):
            mask_block[i * patch_num_row + j][i:i + effective_kernel_size, j:j + effective_kernel_size] = False

    mask_block = mask_block[:, padding:-padding, padding:-padding].reshape(patch_num_row * patch_num_row, patch_num_row * patch_num_row)  # [2560,2560]
    return mask_block

class NearbyAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., patch_num_row=16, kernel_size=5):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads // mpu.get_tensor_model_parallel_world_size()
        self.scale = dim_head ** -0.5

        self.nb_mask = generate_nearby_mask(patch_num_row=patch_num_row, kernel_size=kernel_size)
        self.nb_mask = F.pad(self.nb_mask, (1, 0, 1, 0), value=False)
        i, j = self.nb_mask.shape
        self.nb_mask = self.nb_mask.view(1, 1, i, j)

        self.q_linear = mpu.ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False)
        self.k_linear = mpu.ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False)
        self.v_linear = mpu.ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False)

        self.to_out = mpu.RowParallelLinear(
            inner_dim, dim,
            input_is_parallel=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, extra_q=0, extra_k=0):
        b, n, _, h, device = *q.shape, self.heads, q.device

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      [self.q_linear(q)[0], self.k_linear(k)[0], self.v_linear(v)[0]])

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        mask_value = max_neg_value(dots)

        dots.masked_fill_(self.nb_mask.to(device), mask_value)
        if mask is not None:
            b, i, j = mask.shape
            mask = mask.view(b, 1, i, j)
            dots.masked_fill_(mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)[0]
        out = self.dropout(out)
        return out

# moved and modified from DALLE_pytorch
class ParallelAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.q_linear = mpu.ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False)
        self.k_linear = mpu.ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False)
        self.v_linear = mpu.ColumnParallelLinear(
            dim, inner_dim, bias=False, gather_output=False)

        self.to_out = mpu.RowParallelLinear(
            inner_dim, dim,
            input_is_parallel=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        b, n, _, h, device = *q.shape, self.heads, q.device

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      [self.q_linear(q)[0], self.k_linear(k)[0], self.v_linear(v)[0]])

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        mask_value = max_neg_value(dots)

        if mask is not None:
            b, i, j = mask.shape
            mask = mask.view(b, 1, i, j)
            dots.masked_fill_(mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)[0]
        out = self.dropout(out)
        return out


class LayerScale(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)

    def forward(self, x):
        return x * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class ParallelFeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super().__init__()
        # bias for column / row parallel is enabled by default
        self.dense_h_to_8h = mpu.ColumnParallelLinear(
            dim, dim * mult * 2,
            gather_output=False
        )
        self.activation = nn.Sequential(
            GEGLU(),
            nn.Dropout(dropout)
        )
        self.dense_4h_to_h = mpu.RowParallelLinear(
            dim * mult, dim,
            input_is_parallel=True
        )

    def forward(self, x):
        out = self.dense_h_to_8h(x)[0]
        out = self.activation(out)
        out = self.dense_4h_to_h(out)[0]
        return out

class ClipTextEncoder(nn.Module):
    def __init__(self, model_path, dim):
        super(ClipTextEncoder, self).__init__()
        model, _ = clip.load(model_path, device='cpu')
        self.token_embedding = copy.deepcopy(model.token_embedding)
        self.positional_embedding = copy.deepcopy(model.positional_embedding)
        self.transformer = copy.deepcopy(model.transformer)
        self.ln_final = copy.deepcopy(model.ln_final)
        self.cond_emb = nn.Linear(512, dim)

    def forward(self, cond):
        cond = self.token_embedding(cond)  # [batch_size, n_ctx, d_model]
        cond = cond + self.positional_embedding
        cond = cond.permute(1, 0, 2)  # NLD -> LND
        cond = self.transformer(cond)
        cond = cond.permute(1, 0, 2)  # LND -> NLD
        cond = self.ln_final(cond)
        outputs = self.cond_emb(cond)  # 512 -> dim
        return outputs

class SingleTransformer(nn.Module):
    def __init__(self, attention, attention_cond, ff, dim, depth):
        super().__init__()
        self.atten_norm = nn.LayerNorm(dim)
        self.attention = attention
        self.attention_scale = LayerScale(dim, depth)

        if attention_cond is not None:
            self.atten_norm_cond = nn.LayerNorm(dim)
            self.attention_cond = attention_cond
            self.attention_scale_cond = LayerScale(dim, depth)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = ff
        self.ff_scale = LayerScale(dim, depth)

    def forward(self, x, mask, cond=None, mask_cond=None):
        # attention
        att = self.atten_norm(x)
        att = self.attention(att, att, att, mask)
        att = self.attention_scale(att)
        x = x + att

        # attention_condition
        if cond is not None:
            att = self.atten_norm_cond(x)
            att = self.attention_cond(att, cond, cond, mask_cond)
            att = self.attention_scale_cond(att)
            x = x + att

        # feedforward
        ff = self.ff_norm(x)
        ff = self.ff(ff)
        ff = self.ff_scale(ff)
        ff = x + ff
        return ff

class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            cond=True,
            heads=8,
            dim_head=64,
            ff_mult=4,
            attn_dropout=0.,
            ff_dropout=0.,
            args=None,
            attn_types=None,
            patch_size=16,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.args = args
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for ind, attn_type in zip(range(depth), attn_type_layer):
            if attn_type == 'full':
                attn_class = ParallelAttention
            elif attn_type == 'nearby':
                attn_class = partial(NearbyAttention, patch_num_row=self.args.img_size // patch_size, kernel_size=self.args.kernel_size)                 
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')
            attn_cond_class = ParallelAttention
            self.layers.append(SingleTransformer(
                attn_class(dim, heads=heads,
                           dim_head=dim_head, dropout=attn_dropout),
                attn_cond_class(dim, heads=heads,
                                dim_head=dim_head, dropout=attn_dropout) if cond else None,
                ParallelFeedForward(dim, mult=ff_mult, dropout=ff_dropout),
                dim, ind + 1
            ))

    def forward(self, x, mask=None, cond=None, mask_cond=None):
        for lid in range(len(self.layers)):
            layer = self.layers[lid]
            x = mpu.checkpoint(layer, x, mask, cond, mask_cond)
        return x


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        # 获取visual tokenizer
        self.vae = DFVQGAN8192()

        # 冻结 visual_tokenizer的参数
        for n, p in self.vae.named_parameters():
            p.requires_grad = False

        dim = self.args.dim
        self.encoder = ClipTextEncoder(model_path=os.path.join(BasicArgs.root_dir, "CLIP", "ViT-B-32.pt"), dim=dim)
        self.vae_w = self.args.img_size // self.args.vae_patch_size
        self.vae_h = self.args.img_size // self.args.vae_patch_size
        self.vit_w = self.args.img_size // self.args.vit_patch_size
        self.vit_h = self.args.img_size // self.args.vit_patch_size
        self.image_seq_len_vae = (self.args.img_size // self.args.vae_patch_size) ** 2
        self.image_seq_len_vit = (self.args.img_size // self.args.vit_patch_size) ** 2
        self.image_seq_len = (self.args.img_size // self.args.vae_patch_size) ** 2

        # to make embedding align with tensor model para#
        self.num_image_tokens = self.args.vqvae_vocab_size
        self.vae_emb = nn.Embedding.from_pretrained(copy.deepcopy(self.vae.model.quantize.embedding.weight),
                                                    freeze=False)
        self.image_emb = nn.Linear(self.vae_emb.embedding_dim, dim)
        self.patch_emb = nn.Conv2d(
            3,
            dim,
            kernel_size=self.args.vit_patch_size,
            stride=self.args.vit_patch_size,
        )

        self.image_bos_emb = nn.Parameter(torch.randn(1, dim))
        self.image_msk_emb = nn.Parameter(torch.randn(1, dim))
        #self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(self.image_seq_len,))
        self.image_pos_emb_vae = AxialPositionalEmbedding(dim,
                                                      axial_shape=(1, self.vae_h, self.vae_w))
        self.image_pos_emb_vit = AxialPositionalEmbedding(dim,
                                                      axial_shape=(1, self.vit_h, self.vit_w))

        self.transformer_dec = Transformer(
            dim=dim,
            cond=True,
            depth=args.dec_depth,
            heads=args.heads,
            dim_head=args.dim_head,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            args=self.args,
            attn_types=self.args.attn_types_dec,
            patch_size=self.args.vae_patch_size,
        )
        self.transformer_enc = Transformer(
            dim=dim,
            cond=False,
            depth=args.enc_depth,
            heads=args.heads,
            dim_head=args.dim_head,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            args=self.args,
            attn_types=self.args.attn_types_enc,
            patch_size=self.args.vae_patch_size,
        )
        self.transformer_pth = Transformer(
            dim=dim,
            cond=False,
            depth=args.enc_depth,
            heads=args.heads,
            dim_head=args.dim_head,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            args=self.args,
            attn_types=self.args.attn_types_enc,
            patch_size=self.args.vit_patch_size,
        )

        self.to_logits_img = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )

        self.ignore_index = args.ignore_index
        self.clip_sim = CLIPSimilarity(model_filename=os.path.join(BasicArgs.root_dir, "CLIP", "ViT-B-32.pt"))
        for n, p in self.clip_sim.named_parameters():
            p.requires_grad = False
        # self.aus = nn.Linear(120, 33)
        # self.dummy_param = nn.Parameter(torch.empty(0))

    def load_state_dict(self, state):
        if self.args.load_nuwa:
            new_state = OrderedDict()
            for key in state:
                if "transformer." in key and "clip_sim." not in key and "encoder." not in key:
                    new_key = key.replace("transformer.", "transformer_dec.")
                    new_state[new_key] = state[key]
            super(Net, self).load_state_dict(new_state, strict=False)
        else:
            super(Net, self).load_state_dict(state)

    def forward(self, inputs):
        #torch.cuda.empty_cache()
        self.vae.eval()
        device = self.vae_emb.weight.device  # TODO Never use dummy_param, bugs for unused params of dddp.
        image = inputs.get('label_imgs', None)
        if self.training:
            outputs = self.forward_(inputs, return_loss=True)

        if not self.training:
            temperature = self.args.temperature
            image_seq_len = self.image_seq_len
            outputs = {}

            gen_images = []
            gen_seq = []
            #zero_mask = torch.zeros_like(inputs["vision_mask"]).bool()
            gt_image_base, gt_hm, gt_hms = self.vae.get_codebook_indices(inputs["label_imgs"], inputs["vision_mask"].bool())
            _image_base, _hm, _hms = self.vae.get_codebook_indices(inputs["masked_image"], inputs["vision_mask"].bool())
            for try_it in range(self.args.sample_K):
                if not getattr(args, 'do_train', False):
                    # TODO reset seed to let the model generate more samples.
                    seed = torch.seed()
                    os.environ['PYHTONHASHSEED'] = str(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                # TODO
                text_mask, vae_mask = inputs["text_mask"].clone(), inputs["vae_mask"].clone()
                text, image_ = inputs["input_ids"].clone(), inputs["masked_image"].clone()
                masked_image = inputs["masked_image_blackmask"].clone()
                image_base = _image_base.clone()
                B, W, H = vae_mask.shape
                B = text.size(0)
                image_base = image_base * ~vae_mask.bool().reshape(B, -1)
                # video = []
                for cur_len in tqdm(range(image_seq_len), miniters=5):
                    flatten_vae_mask = vae_mask.reshape(B, -1)
                    use_pred = flatten_vae_mask[:, cur_len].bool()
                    if use_pred.max().item():
                        new_inputs = {
                            'input_ids': text, 'label_imgs': image_,
                            "text_mask": text_mask, "vae_mask": vae_mask,
                            "masked_image": masked_image,
                        }
                        cur_outputs = self.forward_(new_inputs, return_loss=False, image_base=image_base)

                        logits = cur_outputs["logits_seq"][:,cur_len].reshape(B, self.num_image_tokens)
                        filtered_logits = top_k(logits, tk=self.args.tk)
                        probs = F.softmax(filtered_logits / temperature, dim=-1)
                        sample = torch.multinomial(probs, 1)
                        image_base[:, cur_len] = image_base[:, cur_len] * ~use_pred + sample * use_pred
                    
                    #flatten_vae_mask[:, cur_len] = 0
                    #vae_mask = flatten_vae_mask.reshape(B, W, H)

                img_seq = image_base
                images = self.vae.decode(img_seq, _hm, _hms)

                gen_images.append(rearrange(images, '(b l) c w h -> b l c w h', b=B))
                gen_seq.append(rearrange(img_seq, '(b l) d -> b l d', b=B))
            
            gt_images = self.vae.decode(gt_image_base, gt_hm, gt_hms)
            gen_gt_images = [rearrange(gt_images, '(b l) c w h -> b l c w h', b=B)]
            outputs["logits_gt_vae_image"] = torch.stack(gen_gt_images, dim=1)

            outputs["loss_total"] = torch.Tensor([0]).to(device)
            outputs['logits_img'] = torch.stack(gen_images, dim=1)
            outputs['logits_seq'] = torch.cat(gen_seq, dim=1)

            best_n = getattr(args, 'best_n', 1)
            b, k, l, c, w, h = outputs['logits_img'].size()
            my_text = inputs['input_text']
            my_image = rearrange(outputs['logits_img'], 'b k l c w h->(b k l) c w h')
            sim_matrix = self.clip_sim(my_text, my_image, batch_size=b)  # [b, 1, kl]
            sim_matrix_arrange = rearrange(sim_matrix, 'b () (k l)-> b k l', k=k).mean(axis=2)  # [b, k]

            sim_matrix_sorted, idx = torch.sort(sim_matrix_arrange, dim=1, descending=True)
            best_idx = idx[:, :best_n]
            best_imgs = outputs['logits_img'].view(b * k, l, c, w, h)[best_idx.view(b * k)].view(b, k, l, c, w, h)
            best_seqs = outputs['logits_seq'].view(b * k, l, -1)[best_idx.view(b * k)].view(b, k, l, -1)
            # Override logits_img and logits_seq to sorted version.
            outputs['logits_img'] = best_imgs  # [b, k, l, c, w, h]
            outputs['logits_seq'] = best_seqs  # [b, k, l, d]
            outputs['metric_clip'] = sim_matrix_sorted[:, 0].mean()
            outputs['logits_text'] = my_text
            outputs['logits_label_class'] = inputs["mask_class"]

            my_gt_image = inputs['label_imgs']
            sim_matrix_gt = self.clip_sim(my_text, my_gt_image, batch_size=b)  # [b, 1, l]
            sim_matrix_gt_arrange = rearrange(sim_matrix_gt, 'b () l-> b l', l=l).mean(axis=1)  # [b]
            outputs['metric_clip_gt'] = sim_matrix_gt_arrange.mean()
            outputs['metric_relative'] = outputs['metric_clip'] / outputs['metric_clip_gt']

        return outputs

    def forward_(self, inputs, return_loss=True, image_base=None):
        text = inputs['input_ids']

        device = text.device
        B = text.size(0)

        vae_cache_1 = inputs["vae_mask"].eq(1).reshape(B, -1, 1)
        vae_cache_0 = ~vae_cache_1

        tokens_clip = self.encoder(text).to(device)

        if image_base is None:
            image_base, _, _ = self.vae.get_codebook_indices(inputs['label_imgs'], inputs["vision_mask"])
        raw_image = inputs["masked_image"]
        image = image_base.view(B, -1)

        image_target = image * vae_cache_1.view(B, -1) -100 * vae_cache_0.view(B, -1)
        image_target = image_target.long()

        target_ids = self.vae_emb(image)
        image_emb = self.image_emb(target_ids).view(B, -1, self.args.dim)
        image_emb += self.image_pos_emb_vae(image_emb)

        raw_emb = self.patch_emb(raw_image).view(B, -1, self.args.dim)
        raw_emb += self.image_pos_emb_vit(raw_emb)

        tokens_dec = image_emb * vae_cache_1 + self.image_msk_emb * vae_cache_0
        tokens_dec = torch.cat((repeat(self.image_bos_emb, 'n d -> b n d', b=B), tokens_dec), dim=1)

        tokens_enc = image_emb * vae_cache_0 + self.image_msk_emb * vae_cache_1
        tokens_pth = raw_emb

        dec_len = tokens_dec.shape[1]
        enc_len = tokens_enc.shape[1]
        pth_len = tokens_pth.shape[1]
        txt_len = tokens_clip.shape[1]

        task_mask = vae_cache_1.view(B, -1)
        mask_enc_self = None#task_mask.view(B, 1, -1)
        mask_enc_pth_self = None#torch.zeros((B, 1, pth_len), dtype=torch.bool).to(device)
        
        vis_causal = torch.ones(dec_len, dec_len, device=device, dtype=torch.bool).triu_(dec_len - dec_len + 1)
        mask_dec_self = vis_causal.view(1, dec_len, dec_len)
        #mask_dec_cross_img = torch.zeros(B, 1, enc_len + pth_len, device=device).bool()
        #mask_dec_cross_txt = text.eq(0).view(B, 1, -1)
        mask_dec_cross = None#torch.cat((mask_dec_cross_img, mask_dec_cross_txt), dim=-1)
        #mask_dec_cross = torch.cat((mask_dec_cross_img, mask_dec_cross_txt), dim=-1)

        enc_out = self.transformer_enc(tokens_enc, mask=mask_enc_self)
        enc_out_pth = self.transformer_pth(tokens_pth, mask=mask_enc_pth_self)
        tokens_cond = torch.cat((enc_out, enc_out_pth, tokens_clip), dim=-2)
        #tokens_cond = torch.cat((enc_out, tokens_clip), dim=-2)

        dec_out = self.transformer_dec(tokens_dec, mask=mask_dec_self, cond=tokens_cond, mask_cond=mask_dec_cross)
        logits_seq = self.to_logits_img(dec_out[:, :-1, :]).reshape(B, -1, self.num_image_tokens)

        outputs = {}

        if return_loss:
            loss_img = F.cross_entropy(rearrange(logits_seq, 'b n c -> b c n'), image_target)

            outputs['loss_total'] = loss_img
        else:
            outputs['logits_seq'] = logits_seq

        #self.gpu_tracker.track()
        #torch.cuda.empty_cache()
        return outputs

