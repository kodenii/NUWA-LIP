from typing import Tuple
from config import *
from collector.mps_inference import inner_collect_fn, after_collect_fn
from transformers import BertConfig, BertModel, BertTokenizer, DataCollatorForWholeWordMask
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

import os
import random
from torchvision import transforms, utils
from torch.utils.data import Dataset, SequentialSampler, DataLoader, RandomSampler
import torch
import einops
import pickle
import clip

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

torch.multiprocessing.set_sharing_strategy('file_system')

class MaskCoco(Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.root = os.path.join(self.args.root_dir, "dataset/MaskCOCO/eval_visu")
        assert split == "val"

    def __len__(self):
        if getattr(self.args, 'max_eval_samples', None):
            return self.args.max_eval_samples
        else:
            return 5000

    def __getitem__(self, idx):
        idx = idx
        file_name = "{:0>4d}".format(idx)
        full_path_img = os.path.join(self.root, "gt", file_name + ".png")
        full_path_mask = os.path.join(self.root, "mask", file_name + ".pkl")
        full_path_maskimg = os.path.join(self.root, "mask_img", file_name + ".png")
        full_path_guidance = os.path.join(self.root, "guidance", file_name + ".txt")

        img_l = Image.open(full_path_img).convert('RGB')
        img = transforms.ToTensor()(img_l)
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        masked_img_l = Image.open(full_path_maskimg).convert('RGB')
        masked_img = transforms.ToTensor()(masked_img_l)
        masked_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(masked_img)
        with open(full_path_mask, "rb") as reader:
            mask = pickle.load(reader)
        with open(full_path_guidance) as reader:
            caption = reader.readlines()[0]
        mask_pattern = repeat(mask.eq(0), "h w -> c h w", c=3)
        masked_img_blackmask = masked_img * mask_pattern

        outputs = {
            'input_text': caption,
            'label_imgs': img,
            "masked_image": masked_img_blackmask,
            "masked_image_blackmask": masked_img_blackmask,
            "vision_mask": mask,
            "mask_class": "dataset",
            "file_name": file_name,
        }

        return outputs

class CombinedDataset(Dataset):
    """All supported retrieval dataset."""

    def __init__(self, args, split='train'):
        """
        Args:
        """
        assert split == "val"
        self.args = args
        self.split = split

        args_maskcoco = copy.deepcopy(args)

        self.dataset_maskcoco = MaskCoco(args_maskcoco, split)
        self.len_maskcoco = len(self.dataset_maskcoco)

        logger.warning(f'Dataset size MaskCOCO:{self.len_maskcoco}')

    def __len__(self):
        if getattr(self.args, 'max_eval_samples', None):
            return min(self.args.max_eval_samples, self.len_maskcoco)
        else:
            return self.len_maskcoco

    def __getitem__(self, idx):
        if idx < self.len_maskcoco:
            outputs = self.dataset_maskcoco[idx]
            outputs['data_source'] = 'maskcoco'
        return outputs

class BaseDataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        assert split == "val"
        self.args = args
        self.combined_dataset = CombinedDataset(args, split)
        self.image_size = (self.args.img_size, self.args.img_size)
        self.vae_size = (self.args.img_size // self.args.patch_size, self.args.img_size // self.args.patch_size)


    def _tokenize(self, sent):
        out = clip.tokenize(sent)
        return out[0]

    def _get_vmlm(self, index):
        outputs = self.combined_dataset.__getitem__(index)
        outputs['input_ids'] = self._tokenize(outputs["input_text"])
        outputs["text_mask"] = torch.zeros(size=[self.args.max_source_len + 1], dtype=torch.bool)
        outputs["vae_mask"] = einops.reduce(outputs["vision_mask"], "(h h1) (w w1) -> h w", reduction="max", h1=self.args.patch_size, w1=self.args.patch_size)
        return outputs

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, index):
        outputs = self._get_vmlm(index)
        return outputs