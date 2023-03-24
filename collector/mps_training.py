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

    big_list = []
    for sample in collect_list:
        if 'label_imgs' in sample:
            big_list.append(sample['label_imgs'])
            big_list.append(sample['masked_image'])
        big_list.extend(sample['logits_img'][0])
        big_list.extend(sample["logits_gt_vae_image"][0])
    big_tensor = torch.stack(big_list, dim=0)
    data2file(big_tensor, os.path.join(eval_log_dir, f'output{splice}.png'), nrow=len(collect_list),
              normalize=True,
              range=(-1, 1),override=True)
    data2file([e['logits_text'] for e in collect_list], os.path.join(eval_log_dir, f'text{splice}.txt'),
              override=True)
    if 'logits_label_class' in collect_list[0]:
        data2file([e['logits_label_class'] for e in collect_list], os.path.join(eval_log_dir, f'class{splice}.txt'),
                override=True)

    #data2file([e['logits_seq'].cpu().data.numpy().tolist() for e in collect_list],
    #          os.path.join(eval_log_dir, f'text_seq{splice}.txt'), override=True)
