from config import *
from typing import Union, Optional, List, Tuple, Text, BinaryIO

inner_collect_fn = default_inner_collect_fn

def after_collect_fn(collect_list, eval_meter, log_dir, epoch):
    rank = int(os.getenv('RANK', '-1'))
    if rank == -1:
        splice = ''
    else:
        splice = '_' + str(rank)
    if epoch == -1:
        eval_log_dir = os.path.join(log_dir, 'eval_visu')
    else:
        eval_log_dir = os.path.join(log_dir, 'train_%05d' % (epoch + 1))
    logger.warning(eval_log_dir)

    if rank in [-1, 0]:
        ensure_dirname(eval_log_dir)

    for idx, sample in enumerate(collect_list):
        img = transforms.Resize(256)(sample['logits_img'][0])
        file_name = sample['logits_name']
        data2file(img, os.path.join(eval_log_dir, file_name + ".png"), nrow=1,
              normalize=True,
              range=(-1, 1), override=True)


    #data2file([e['logits_seq'].cpu().data.numpy().tolist() for e in collect_list],
    #          os.path.join(eval_log_dir, f'text_seq{splice}.txt'), override=True)
