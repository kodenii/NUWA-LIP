from config import *
from collector.dfvqgan import inner_collect_fn, after_collect_fn

import einops

class MaskingGenerator:
    def __init__(
            self, input_size, max_num_patches, min_num_patches,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width

        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        #for attempt in range(1):
        num = 0
        for attempt in range(5):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
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
                if num + h * w - num_masked <= max_mask_patches:
                    mask[top: top + h, left: left + w] = 1
                    num = num + h * w - num_masked
        return num

    def __call__(self):
        mask = torch.zeros(size=self.get_shape(), dtype=torch.int)

        self._mask(mask, self.max_num_patches)

        return mask

class ImageNetDataset(Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        with open(os.path.join(args.root_dir, "dataset", "imagenet", "imagenet_datasets", 'imagenet_train.txt'), 'r') as f:
            tmp = f.readlines()
        self.filelist = []
        for filename in tmp:
            self.filelist.append(os.path.join(args.root_dir, "dataset", "imagenet", "imagenet_datasets", filename.strip('\n')))
        if split == 'train':
            if getattr(self.args, 'train_samples', False):
                self.filelist = self.filelist[:args.max_samples]
            else:
                self.filelist = self.filelist[:-32]
        elif split == 'val':
            self.filelist = self.filelist[-32:]

        if hasattr(args, 'transform'):
            self.transform = args.transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size, scale=(0.65, 1.), ratio=(1., 1.)),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.05, contrast=0.25, saturation=0.25, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, i):
        fileroot = self.filelist[i]
        image = Image.open(fileroot).convert("RGB")
        image = self.transform(image)
        return {"input_img": image}



class CombinedDataset(Dataset):
    """All supported retrieval dataset."""

    def __init__(self, args, split='train'):
        """
        Args:
        """
        self.args = args
        self.split = split

        args_img = copy.deepcopy(args)

        self.dataset_img = ImageNetDataset(args_img, split)
        self.len_img = len(self.dataset_img)

        logger.warning(f'Dataset size ImageNet:{self.len_img}')

    def __len__(self):
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return min(self.args.max_train_samples, self.len_img)
            else:
                return self.len_img  # Forged quantity
        else:  # eval
            if getattr(self.args, 'max_eval_samples', None):
                return min(self.args.max_eval_samples, self.len_img)
            else:
                return self.len_img

    def __getitem__(self, idx):
        if idx < self.len_img:
            outputs = self.dataset_img[idx]
            outputs['data_source'] = 'img'
        return outputs

class BaseDataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        self.combined_dataset = CombinedDataset(args, split)
        self.accumulated_ratio = []
        self.image_size = (self.args.img_size, self.args.img_size)
        self.vae_size = (self.args.img_size // self.args.patch_size, self.args.img_size // self.args.patch_size)
        self.ramdom_vmasker = MaskingGenerator(self.image_size, \
            int(self.image_size[0] * self.image_size[1] * self.args.max_vmlm_ratio), \
                int(self.image_size[0] * self.image_size[1] * self.args.min_vmlm_ratio))

    def _mask_vision(self, outputs):
        if self.split == "train":
            r = random.random()
            r2 = random.random()
        else:
            r = 0
            r2 = self.args.none_decoder_ratio
        if r < self.args.random_mask_ratio:
            mask = self.ramdom_vmasker()
            mask_class = "random"
            if r2 < self.args.none_decoder_ratio:
                decoder_type = "none"
            else:
                decoder_type = "extra"
        elif r < self.args.random_mask_ratio + self.args.full_mask_ratio:
            mask = torch.zeros(size=self.image_size, dtype=torch.int)
            mask_class = "none"
            decoder_type = "none"
        else:
            mask = torch.ones(size=self.image_size, dtype=torch.int)
            mask_class = "full"
            if r2 < self.args.none_decoder_ratio:
                decoder_type = "none"
            else:
                decoder_type = "extra"
        return mask, mask_class, decoder_type

    def _get_sample(self, index):
        outputs = self.combined_dataset.__getitem__(index)
        outputs["label_imgs"] = outputs["input_img"]
        outputs["vision_mask"], outputs["mask_class"], outputs["decoder_type"] = self._mask_vision(outputs)
        mask_pattern = repeat(outputs["vision_mask"].eq(0), "h w -> c h w", c=3)
        outputs["vae_mask"] = einops.reduce(outputs["vision_mask"], "(h h1) (w w1) -> h w", reduction="max", h1=self.args.patch_size, w1=self.args.patch_size)
        outputs["masked_image"] = outputs["label_imgs"] * mask_pattern
        outputs["split"] = self.split
        return outputs

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, index):
        outputs = self._get_sample(index)
        return outputs