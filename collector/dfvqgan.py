from config import *

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

    input_img_list = [e['label_imgs'].cpu() for e in collect_list]
    input_masked_img_list = [e['masked_image'].cpu() for e in collect_list]
    logits_img_list = [e['logits_img'].cpu() for e in collect_list]
    logits_imgmask_list_more = [e['logits_imgmaskmoreinfo'].cpu() for e in collect_list]
    logits_imgmask_list_no = [e['logits_imgmasknoinfo'].cpu() for e in collect_list]


    big_tensor = torch.stack(input_img_list + input_masked_img_list + logits_img_list + logits_imgmask_list_more + logits_imgmask_list_no, dim=0)
    data2file(big_tensor, os.path.join(eval_log_dir, f'output{splice}.png'), nrow=len(input_img_list),
              normalize=True,
              range=(-1, 1), override=True)
