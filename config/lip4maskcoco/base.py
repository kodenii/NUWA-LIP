import sys

sys.path.append('../..')
from config import *
from config.lip4maskcoco import *
from config.mps4coco.base import Net as Painter

@iterable_class
class Args(BasicArgs):
    pretrained_model = os.path.join(BasicArgs.root_dir, "checkpoint/pretrained/epoch66-chunk-0.pth")
    task_name, method_name, log_dir = BasicArgs.get_log_dir(__file__)
    max_train_samples = None
    max_eval_samples = None
    max_source_len = 35
    img_size = 256
    vqvae_vocab_size = 8192

    load_nuwa = False
    adapt_load = True

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
    learning_rate = 1.5e-3
    epochs = 5000
    seed = 42
    num_workers = 1
    eval_step = 50
    save_step = 50

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
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_prompt = transforms.Compose([
        transforms.ToTensor(),
    ])

args = Args()

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.painter = Painter(args)
        self.painter.load_state_dict(torch.load(os.path.join(self.args.root_dir, self.args.ckpt), map_location="cpu")["model"])

    def forward(self, inputs):
        outputs = self.painter(inputs)
        outputs["logits_name"] = inputs["file_name"]
        return outputs

