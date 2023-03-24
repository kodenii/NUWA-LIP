from typing import Tuple
from config import *
from collector.mps_training import inner_collect_fn, after_collect_fn

import os
import random
from torchvision import transforms, utils
from torch.utils.data import Dataset, SequentialSampler, DataLoader, RandomSampler
import torch
import einops
import clip

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

torch.multiprocessing.set_sharing_strategy('file_system')

class MaskingGenerator:
    def __init__(
            self, input_size, max_num_patches, min_num_patches, 
            max_per_num, min_per_num, attempt,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width

        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches

        self.max_per_num = max_per_num
        self.min_per_num = min_per_num
        self.attempt = attempt

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask):
        num = 0
        for _ in range(self.attempt):
            target_area = random.uniform(self.min_per_num, self.max_per_num)
            #target_area = 49
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            #aspect_ratio = 1
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                attempt_num = num + h * w - num_masked
                if attempt_num <= self.max_num_patches:
                    mask[top: top + h, left: left + w] = 1
                    num = attempt_num
                    if attempt_num > self.min_num_patches:
                        break
        return num

    def __call__(self):
        mask = torch.zeros(size=self.get_shape(), dtype=torch.int)

        self._mask(mask)

        return mask

class CC(Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.split = split
        self.name = 'cc'
        self.prefix = "%s_%s_" % (self.name, self.split)
        self.is_master_thread = 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] in ['-1', '0']
        self.local_root_dir = getattr(args, 'local_root_dir', False)
        self.basic_root_dir = BasicArgs.root_dir
        dataset_name = self.name

        if split == 'train':
            file_root = os.path.join(self.basic_root_dir, 'dataset/cc/utils/Train_GCC-training.tsv')
            self.root = os.path.join(self.basic_root_dir, 'dataset/cc/train_image')
            exist_filename = os.path.join(self.basic_root_dir, 'dataset/cc/train_image_exist.txt')
        elif self.split == 'val':
            file_root = os.path.join(self.basic_root_dir, 'dataset/cc/utils/Validation_GCC-1.1.0-Validation.tsv')
            self.root = os.path.join(self.basic_root_dir, 'dataset/cc/val_image')
            exist_filename = os.path.join(self.basic_root_dir, 'dataset/cc/val_image_exist.txt')

        else:
            raise ValueError('Not supported split %s.' % self.split)
        exist_data = set(file2data(exist_filename))
        exist_data = set([int(f.split(os.sep)[-1][:-4]) for f in exist_data])
        self.data = [{'image_name': "%08d.jpg" % i, 'captions': [e.split('\t')[0]]} for i, e in enumerate(file2data(file_root, type='txt')) if i in exist_data]

        if hasattr(args, 'transform'):
            self.transform = args.transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.), ratio=(1., 1.)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.05, contrast=0.25, saturation=0.25, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return self.args.max_train_samples
            else:
                return len(self.data)
        else:
            if getattr(self.args, 'max_eval_samples', None):
                return self.args.max_eval_samples
            else:
                return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        filename, captions = data['image_name'], data['captions']
        filename = os.path.join(self.root, filename)

        img = Image.open(filename).convert('RGB')

        img = self.transform(img)

        if not getattr(self.args, 'do_train', False):
            caption = captions[0]
        else:
            caption = random.sample(captions, 1)[0]

        outputs = {'input_text': caption, 'label_imgs': img}
        return outputs

class CombinedDataset(Dataset):
    """All supported retrieval dataset."""

    def __init__(self, args, split='train'):
        """
        Args:
        """
        self.args = args
        self.split = split

        args_cc = copy.deepcopy(args)

        self.dataset_cc = CC(args_cc, split)
        self.len_cc = len(self.dataset_cc)

        logger.warning(f'Dataset size GCC:{self.len_cc}')

    def __len__(self):
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return min(self.args.max_train_samples, self.len_cc)
            else:
                return self.len_cc  # Forged quantity
        else:  # eval
            if getattr(self.args, 'max_eval_samples', None):
                return min(self.args.max_eval_samples, self.len_cc)
            else:
                return self.len_cc

    def _get_caption(self, idx):
        if idx < self.len_cc:
            outputs = self.dataset_cc._get_caption(idx)
        return outputs

    def __getitem__(self, idx):
        if idx < self.len_cc:
            outputs = self.dataset_cc[idx]
            outputs['data_source'] = 'cc'
        return outputs

class BaseDataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.combined_dataset = CombinedDataset(args, split)
        self.image_size = (self.args.img_size, self.args.img_size)
        self.vae_size = (self.args.img_size // self.args.vae_patch_size, self.args.img_size // self.args.vae_patch_size)
        self.ramdom_vmasker = MaskingGenerator(self.image_size, \
            int(self.image_size[0] * self.image_size[1] * self.args.max_vmlm_ratio), \
                int(self.image_size[0] * self.image_size[1] * self.args.min_vmlm_ratio), \
                    int(self.image_size[0] * self.image_size[1] * self.args.max_per_mask_ratio), \
                        int(self.image_size[0] * self.image_size[1] * self.args.min_per_mask_ratio), self.args.attempt)


    def _tokenize(self, sent):
        out = clip.tokenize(sent, truncate=True)
        return out[0]

    def _mask_vision(self, outputs):
        #r = random.random()
        mask = self.ramdom_vmasker()
        mask_class = "random"
        if getattr(self.args, "full_mask", False):
            mask = torch.ones(size=self.image_size, dtype=torch.int)
            mask_class = "full"
        return mask, mask_class

    def _get_vmlm(self, index):
        outputs = self.combined_dataset.__getitem__(index)
        outputs['input_ids'] = self._tokenize(outputs["input_text"])
        outputs["vision_mask"], outputs["mask_class"] = self._mask_vision(outputs)
        outputs["text_mask"] = torch.zeros(size=[self.args.max_source_len + 1], dtype=torch.bool)
        mask_pattern = repeat(outputs["vision_mask"].eq(0), "h w -> c h w", c=3)
        outputs["vae_mask"] = einops.reduce(outputs["vision_mask"], "(h h1) (w w1) -> h w", reduction="max", h1=self.args.vae_patch_size, w1=self.args.vae_patch_size)
        outputs["masked_image"] = outputs["label_imgs"] * mask_pattern
        outputs["masked_image_blackmask"] = outputs["label_imgs"] * mask_pattern
        outputs["task"] = "vmlm"
        return outputs

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, index):
        outputs = self._get_vmlm(index)
        return outputs