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

class MSCoco(Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.seg_prompt = getattr(args, 'seg_prompt', False)
        self.box_prompt = args.object_mask_ratio > 0
        self.split = split
        self.name = 'mscoco'
        self.prefix = "%s_%s_" % (self.name, self.split)
        self.is_master_thread = 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] in ['-1', '0']
        self.local_root_dir = getattr(args, 'local_root_dir', False)
        self.basic_root_dir = BasicArgs.root_dir
        self.image_size = (self.args.img_size, self.args.img_size)
        self.ramdom_vmasker = MaskingGenerator(self.image_size, \
            int(self.image_size[0] * self.image_size[1] * self.args.max_vmlm_ratio), \
                int(self.image_size[0] * self.image_size[1] * self.args.min_vmlm_ratio), \
                    int(self.image_size[0] * self.image_size[1] * self.args.max_per_mask_ratio), \
                        int(self.image_size[0] * self.image_size[1] * self.args.min_per_mask_ratio), self.args.attempt)
        dataset_name = self.name
        if split == 'train':
            self.data = file2data(os.path.join(self.basic_root_dir, 'dataset/mscoco/train_data.json'))
            self.root = os.path.join(self.basic_root_dir, 'dataset/mscoco/train2017')
            if self.seg_prompt:
                self.data_seg = [{'image_name': e['image_name'].replace('jpg', 'png'), 'captions': e['captions']} for e
                                 in self.data]
                self.root_seg = os.path.join(self.basic_root_dir, 'dataset/mscoco/train2017_stuff')

            if self.box_prompt:
                data_box_raw = file2data(os.path.join(self.basic_root_dir, 'dataset/mscoco/stuff_train2017.json'))
                data_box_raw_dict = groupby(data_box_raw['images'], lambda x: x['file_name'])
                self.categories = {x["id"]: x["name"] for x in data_box_raw["categories"]}
                self.data_box = {k: {'boxes': [e['bbox'] for e in v], 'classes': [e['category_id'] for e in v], 'width': data_box_raw_dict[k][0]['width'],
                                     'height': data_box_raw_dict[k][0]['height']} for k, v in
                                 groupby(data_box_raw['annotations'], lambda x: "%012d.jpg" % x['image_id']).items()}


        elif self.split == 'val':
            self.data = file2data(os.path.join(self.basic_root_dir, 'dataset/mscoco/val_data.json'))
            self.root = os.path.join(self.basic_root_dir, 'dataset/mscoco/val2017')
            if self.seg_prompt:
                self.data_seg = [{'image_name': e['image_name'].replace('jpg', 'png'), 'captions': e['captions']} for e
                                 in self.data]
                self.root_seg = os.path.join(self.basic_root_dir, 'dataset/mscoco/val2017_stuff')

            if self.box_prompt:
                data_box_raw = file2data(os.path.join(self.basic_root_dir, 'dataset/mscoco/stuff_val2017.json'))
                data_box_raw_dict = groupby(data_box_raw['images'], lambda x: x['file_name'])
                self.categories = {x["id"]: x["name"] for x in data_box_raw["categories"]}
                self.data_box = {k: {'boxes': [e['bbox'] for e in v], 'width': data_box_raw_dict[k][0]['width'],
                                     'height': data_box_raw_dict[k][0]['height'],
                                     'classes': [e['category_id'] for e in v]} for k, v in
                                 groupby(data_box_raw['annotations'], lambda x: "%012d.jpg" % x['image_id']).items()}

        else:
            raise ValueError('Not supported split %s.' % self.split)

        if hasattr(args, 'transform'):
            self.transform = args.transform
            self.transform_prompt = getattr(args, 'transform_prompt', None)
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.7, 1.), ratio=(1., 1.)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.05, contrast=0.25, saturation=0.25, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.transform_prompt = transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.7, 1.), ratio=(1., 1.)),
                transforms.RandomHorizontalFlip(p=0.1),
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

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)

    def __getitem__(self, idx):
        data = self.data[idx]
        filename, captions = data['image_name'], data['captions']
        filename = os.path.join(self.root, filename)
        if self.seg_prompt:
            data_seg = self.data_seg[idx]
            filename_prompt, captions_prompt = data_seg['image_name'], data_seg['captions']
            filename_prompt = os.path.join(self.root_seg, filename_prompt)

        img = Image.open(filename).convert('RGB')
        state = torch.get_rng_state()
        img = self.augmentation(img, self.transform, state)
        if self.seg_prompt:
            imgs_seg = Image.open(filename_prompt)
            imgs_seg = np.array(imgs_seg).astype(np.uint8)
            imgs_seg = imgs_seg + 1
            imgs_seg = Image.fromarray(imgs_seg)  # [h,w]
            imgs_seg = self.augmentation(imgs_seg, self.transform_prompt, state)  # [h,w]
            imgs_seg = np.array(imgs_seg).astype(np.uint8)
            n_labels = 183
            flatseg = np.ravel(imgs_seg)  # h*w
            onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)  # [h*w,183]
            onehot[np.arange(flatseg.size), flatseg] = True
            onehot = onehot.reshape(imgs_seg.shape + (n_labels,)).astype(int)
            # [h,w,n_label] -> [n_label,h,w]
            imgs_seg = torch.from_numpy(onehot).type(torch.LongTensor).permute(2, 0, 1)
        if not getattr(self.args, 'do_train', False):
            caption = captions[0]
        else:
            caption = random.sample(captions, 1)[0]

        outputs = {'input_text': caption, 'label_imgs': img}
        if self.seg_prompt:
            outputs['input_seg'] = imgs_seg
        if self.box_prompt:
            try:
                meta = self.data_box[os.path.basename(filename)]
                boxes = meta['boxes']
                combined = list(zip(boxes, meta["classes"]))
                random.shuffle(combined)
                for selected_obj, class_id in combined:
                    class_name = self.categories[class_id]
                    mask_obj = np.zeros((meta['height'], meta['width']), dtype=np.bool)
                    mask_obj[int(selected_obj[1]):int(selected_obj[1] + selected_obj[3]), \
                        int(selected_obj[0]):int(selected_obj[0] + selected_obj[2])] = True
                    mask_image = Image.fromarray(mask_obj)
                    mask_obj = self.augmentation(mask_image, self.transform_prompt, state).bool()
                    if mask_obj.sum() >= int(self.args.img_size * self.args.img_size * self.args.min_vmlm_ratio):
                        break
                outputs['mask_obj'] = mask_obj[0].int()
                outputs["mask_class"] = class_name
            except:
                outputs['mask_obj'] = self.ramdom_vmasker()
                outputs["mask_class"] = "random"

        return outputs

class MSCocoRetrieval(MSCoco):
    """MSCOCO for retrieval"""

    def __init__(self, args, split='train'):
        super().__init__(args, split=split)

    def _get_caption(self, index):
        data = self.data[index]
        captions = data['captions']
        if not getattr(self.args, 'do_train', False):
            caption = captions[0]
        else:
            caption = random.sample(captions, 1)[0]
        return caption

class CombinedDataset(Dataset):
    """All supported retrieval dataset."""

    def __init__(self, args, split='train'):
        """
        Args:
        """
        self.args = args
        self.split = split

        args_mscoco = copy.deepcopy(args)

        self.dataset_coco = MSCocoRetrieval(args_mscoco, split)
        self.len_coco = len(self.dataset_coco)

        logger.warning(f'Dataset size MSCOCO:{self.len_coco}')

    def __len__(self):
        if self.split == 'train':
            if getattr(self.args, 'max_train_samples', None):
                return min(self.args.max_train_samples, self.len_coco)
            else:
                return self.len_coco  # Forged quantity
        else:  # eval
            if getattr(self.args, 'max_eval_samples', None):
                return min(self.args.max_eval_samples, self.len_coco)
            else:
                return self.len_coco

    def _get_caption(self, idx):
        if idx < self.len_coco:
            outputs = self.dataset_coco._get_caption(idx)
        return outputs

    def __getitem__(self, idx):
        if idx < self.len_coco:
            outputs = self.dataset_coco[idx]
            outputs['data_source'] = 'coco'
        return outputs

class BaseDataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.combined_dataset = CombinedDataset(args, split)
        self.image_size = (self.args.img_size, self.args.img_size)
        self.vae_size = (self.args.img_size // self.args.patch_size, self.args.img_size // self.args.patch_size)
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
        outputs["vae_mask"] = einops.reduce(outputs["vision_mask"], "(h h1) (w w1) -> h w", reduction="max", h1=self.args.patch_size, w1=self.args.patch_size)
        outputs["masked_image"] = outputs["label_imgs"] * mask_pattern
        outputs["masked_image_blackmask"] = outputs["label_imgs"] * mask_pattern
        outputs["task"] = "vmlm"
        return outputs

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, index):
        outputs = self._get_vmlm(index)
        return outputs