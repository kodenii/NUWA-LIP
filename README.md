# NÜWA-LIP: Language Guided Image Inpainting with Defect-free VQGAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is for the CVPR 2023 paper: [NÜWA-LIP: Language Guided Image Inpainting with Defect-free VQGAN](about:blank)

If you use any source codes or ideas included in this repository for your work, please cite the following paper.
<pre>
@inproceedings{ni2023nuwa,
 title={N{\"U}WA-LIP: Language Guided Image Inpainting with Defect-free VQGAN},
 author={Ni, Minheng and Li, Xiaoming and Zuo, Wangmeng},
 booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
 year={2023}
}
</pre>

If you have any questions, feel free to email [me](Mailto:mhni@stu.hit.edu.cn).

## Abstract

Language-guided image inpainting aims to fill the defective regions of an image under the guidance of text while keeping the non-defective regions unchanged. 
However, directly encoding the defective images is prone to have an adverse effect on the non-defective regions, giving rise to distorted structures on non-defective parts.
To better adapt the text guidance to the inpainting task, this paper proposes NÜWA-LIP, which involves defect-free VQGAN (DF-VQGAN) and a multi-perspective sequence-to-sequence module (MP-S2S).
To be specific, DF-VQGAN introduces relative estimation to carefully control the receptive spreading, as well as symmetrical connections to protect structure details unchanged.
For harmoniously embedding text guidance into the locally defective regions, MP-S2S is employed by aggregating the complementary perspectives from low-level pixels, high-level tokens as well as the text description.
Experiments show that our DF-VQGAN effectively aids the inpainting process while avoiding unexpected changes in non-defective regions. Results on three open-domain benchmarks demonstrate the superior performance of our method against state-of-the-arts.

## Illustration of Framework

<img src="img/NUWA-LIP.png">

## Environment Setup

We provide a docker image to simplify the environment preparation. You can access the image from [kodenii/nuwalip](https://hub.docker.com/r/kodenii/nuwalip).

To start up the environment, use the following command.

```
nvidia-docker run -it kodenii/nuwalip:v1
```

## Training

### Dataset Preparation

By default, you need to arrange the training dataset following this directory structure:

```
data
|--dataset
   |--imagenet
      |--imagenet_datasets
         |--n01440764
         |--...
         |--imagenet_train.txt
   |--cc
      |--train_image
      |--utils
         |--Train_GCC-training.tsv
         |--Validation_GCC-1.1.0-Validation.tsv
      |--val_image
   |--mscoco
      |--train2017
      |--train2017_stuff
      |--val2017
      |--val2017_stuff
      |--stuff_train2017.json
      |--stuff_val2017.json
      |--train_data.json
      |--val_data.json
```

You may need to obtain [ImageNet](https://www.image-net.org/index.php), [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/), and [MSCOCO](https://cocodataset.org) from their offical websites.

### Training DF-VQGAN

To train DF-VQGAN, you may need at least 16GB VRAM for each node. We suggest training over 10 epochs on the ImageNet dataset.

```
./run.sh -a do_train -ltbs 8 -lebs 1 -cf ./config/dfvq4imagenet/base.py -m megatron -p custom -crd $NODE_NAME -cmp 9500
```

### Pre-training MP-S2S from Scratch

To train MP-S2S, you need to finish the training of DF-VQGAN. As MP-S2S is a large autoregressive model, you need GPU(s) with enough VRAM, such as A100. If you want to train MP-S2S with lower VRAM, please modify the hyper-parameters like layer number.

```
./run.sh -a do_train -ltbs 8 -lebs 1 -cf ./config/mps4cc/base.py -m megatron -p custom -crd $NODE_NAME -cmp 9500
```

### Finetuning MP-S2S from Checkpoints

To finetune MP-S2S, you need to specify the pre-trained weights in the arg of ```./config/mps4coco/base.py```. Then, execute the following command.

```
./run.sh -a do_train -ltbs 8 -lebs 1 -cf ./config/mps4coco/base.py -m megatron -p custom -crd $NODE_NAME -cmp 9500 -ckpt $CHECKPOINT_FILE
```

## Inference

### Dataset Preparation

By default, you need to arrange the training dataset following this directory structure:

```
data
|--dataset
   |--MaskCOCO
      |--eval_visu
         |--gt
         |--guidance
         |--mask
         |--mask_img
   |--MaskFlickr
   |--MaskVG
   |--CustomDataset
```
You can download MaskCOCO, MaskFlickr, and MaskVG from the following links.

|Name |Download Link|
|---------------------|-------------|
|MaskCOCO|[Download](https://minheng.blob.core.windows.net/public/NUWA-LIP/maskcoco.zip?sv=2021-10-04&st=2023-03-21T13%3A46%3A28Z&se=2026-03-22T13%3A46%3A00Z&sr=b&sp=r&sig=IW8CfZ5PHrgwWlp8hMT3v%2FyySVFSB9hEhqTCskhwDW4%3D)|
|MaskFlickr|[Download](https://minheng.blob.core.windows.net/public/NUWA-LIP/maskflickr.zip?sv=2021-10-04&st=2023-03-21T13%3A47%3A12Z&se=2026-03-22T13%3A47%3A00Z&sr=b&sp=r&sig=grsJkrP4hktl0S7Nm6vjfm2RHcXCh6vxJ1MMQYeckJA%3D)|
|MaskVG|[Download](https://minheng.blob.core.windows.net/public/NUWA-LIP/maskvg.zip?sv=2021-10-04&st=2023-03-21T13%3A47%3A30Z&se=2026-03-22T13%3A47%3A00Z&sr=b&sp=r&sig=2V9JqOqjp1KNaDFDWbc1ZJozKFs7Wsq%2F7OKdmp6l3Vc%3D)|

### Inference on Proposed Dataset

To infer the model on our proposed dataset, you can follow this command. You may need to prepare the checkpoint first.

```
./run.sh -a eval_visu -p itp -ltbs 1 -lebs 1 -cf ./config/lip4maskcoco/base.py -m megatron -p custom -crd $NODE_NAME -cmp 9500
```

### Inference on Customized Dataset

We support free-form masks for the customized dataset by default. Each free-form mask is a binary PNG image with black and white pixels to indicate the mask. You can also modify code easily to adapt to other formats. See ```config/lip4custom``` for more details.

## Multi-GPU and Multi-node Support

We assemble the multi-GPU and multi-node support in our framework. By default, all GPUs in a node will be used for training. If you want to train the model with multiple nodes, you need to execute the following command in all nodes at the same time. Please make sure all nodes are under the same network and can access each other.

```
./run.sh -a do_train -ltbs 8 -lebs 1 -cf ./config/dfvq4imagenet/base.py -m megatron -p custom -crd $NODE_NAME1,$NODE_NAME2,$NODE_NAME3,... -cmp 9500
```

## Reproducibility

To help improve the reproducibility of the community, we provide checkpoints of pre-trained models. We trained the model for a longer time to obtain better performance. As the influence of the random seed or machine, we suggest to re-generate more times for each sample, which can improve the visual quality significantly.

|Model |Pretrained Data |Finetuned Data|Download Link|
|--------|-------------------|--------------|-------------|
|DF-VQGAN|ImageNet |(None) |[Download](https://minheng.blob.core.windows.net/public/NUWA-LIP/DF-VQGAN/epoch14-chunk-0.pth?sv=2021-10-04&st=2023-03-24T07%3A51%3A53Z&se=2026-03-25T07%3A51%3A00Z&sr=b&sp=r&sig=Cb%2BnRIKUk4VUTlvydMJDrcEkLK8z1gt4KBBViR%2FA1pU%3D)|
|NÜWA-LIP|Conceptual Captions|(None) |[Download](https://minheng.blob.core.windows.net/public/NUWA-LIP/pretrained/epoch66-chunk-0.pth?sv=2021-10-04&st=2023-03-21T13%3A59%3A51Z&se=2026-03-22T13%3A59%3A00Z&sr=b&sp=r&sig=akOZ%2BlefSxpOJuJxlwYcgaRYevR6j4ptVssdqOMuTJs%3D)|
|NÜWA-LIP|Conceptual Captions|MSCOCO |[Download](https://minheng.blob.core.windows.net/public/NUWA-LIP/finetuned/epoch140-chunk-0.pth?sv=2021-10-04&st=2023-03-21T13%3A59%3A13Z&se=2026-03-22T13%3A59%3A00Z&sr=b&sp=r&sig=4vscJhNskC9IxMFnULfabWTnoI3NzT%2B3meK1s6facCQ%3D)|

