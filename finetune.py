from config import *
import os
import argparse
from torchvision import transforms, utils
from torch.utils.data import Dataset, SequentialSampler, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torchinfo import summary
import megatron
from megatron import mpu
from megatron.initialize import initialize_megatron

#from apex import amp

torch.multiprocessing.set_sharing_strategy('file_system')


def main_worker(local_rank, parsed_args):
    '''
    local_rank = -1 means do no use distributed.
    '''

    if os.path.exists(parsed_args.cf):
        cf = import_filename(parsed_args.cf)
        Net, args, BaseDataset, inner_collect_fn, after_collect_fn = \
            cf.Net, cf.args, cf.BaseDataset, cf.inner_collect_fn, cf.after_collect_fn
    else:
        raise NotImplementedError('Config filename %s does not exist.' % parsed_args.cf)
    # args.printable = (local_rank in [-1, 0])
    args.do_train = parsed_args.do_train
    # Note that tbs is the Global batch size = GPU_PER_NODE * NODE_COUNT * TRAIN_LOCAL_BATCH_SIZE
    if parsed_args.tbs:
        args.train_batch_size = parsed_args.tbs
    # Note that ebs is the Global batch size = GPU_PER_NODE * NODE_COUNT * EVAL_LOCAL_BATCH_SIZE
    if parsed_args.ebs:
        args.eval_batch_size = parsed_args.ebs
    if parsed_args.debug:
        args.debug = parsed_args.debug

    if args.debug:
        logger.warning('================Debugging Activated===================')
        args.num_workers = 0
        args.resume = False
        if local_rank == -1:
            args.train_batch_size = 3
            args.eval_batch_size = 3
        else:
            args.train_batch_size = torch.cuda.device_count() * 2
            args.eval_batch_size = torch.cuda.device_count()
        args.max_video_len = 2
        args.save_step = 2000
        args.eval_step = 2
        args.epochs = 2
        args.max_train_samples = 16
        args.max_eval_samples = 8
        args.load_mem = False
        args.log_dir = os.path.join(os.path.dirname(args.log_dir), 'DEBUG')
        if hasattr(args, 'debug_fn'):
            args = args.debug_fn(args)
    else:
        logger.warning('================Common Running===================')
    if parsed_args.ckpt is not None:
        args.ckpt = parsed_args.ckpt
    args.n_gpu = torch.cuda.device_count()
    if local_rank == -1:  # Do not use distributed training.
        args.rank = -1
        os.environ['RANK'] = os.environ['LOCAL_RANK'] = '-1'
        args.local_train_batch_size = args.train_batch_size
        args.local_eval_batch_size = args.eval_batch_size
    else:  # Use torch.distributed.launch for training
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.nodes = int(args.world_size / args.n_gpu)
        args.node_id = int(args.rank / args.n_gpu)
        if parsed_args.dist:
            logger.info('[node:{}/{} rank:{}/{} local_rank:{}/{}] launches'.format(
                args.node_id, args.nodes, args.rank, args.world_size, args.local_rank, args.n_gpu))
            dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
            args.local_train_batch_size = args.train_batch_size // args.world_size
            args.local_eval_batch_size = args.eval_batch_size // args.world_size
            if not args.debug and args.local_eval_batch_size != 1:
                raise ValueError('We have to let local_eval_batch_size=1 now, to avoid same generation on a batch. Got %s', args.local_eval_batch_size)
        elif parsed_args.megatron:
            logger.info('[node:{}/{} rank:{}/{} local_rank:{}/{} ddp-rank:{} tmp-rank:{} pipe-rank:{}] launches'.format(
                args.node_id, args.nodes, args.rank, args.world_size, args.local_rank, args.n_gpu,
                mpu.get_data_parallel_rank(), mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank()))
            args.local_train_batch_size = args.train_batch_size // mpu.get_data_parallel_world_size()
            args.local_eval_batch_size = args.eval_batch_size // mpu.get_data_parallel_world_size()
            if not args.debug and args.local_eval_batch_size != 1:
                raise ValueError('We have to let local_eval_batch_size=1 now, to avoid same generation on a batch. Got %s', args.local_eval_batch_size)
        else:
            raise ValueError('You must specify distributed training method. Either dist or megatron.')

    # init model
    model = Net(args)
    logger.info('Successfully built model with %s parameters' % get_parameters(model))
    # if os.environ['LOCAL_RANK'] in [-1, 0]:
    #     summary(model)
    if parsed_args.do_train:
        logger.warning("Do training...")
        # Prepare Dataset.
        if parsed_args.megatron and mpu.get_tensor_model_parallel_rank() != 0:
            train_dataset = BaseDataset(args, split='train', is_random=True)
            eval_dataset = BaseDataset(args, split='val', is_random=True)
        else:
            train_dataset = BaseDataset(args, split='train')
            eval_dataset = BaseDataset(args, split='val')

        if parsed_args.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True, )
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False)

        elif parsed_args.megatron:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=mpu.get_data_parallel_world_size(),
                rank=mpu.get_data_parallel_rank(),
                shuffle=True, )
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=mpu.get_data_parallel_world_size(),
                rank=mpu.get_data_parallel_rank(),
                shuffle=False,
            )
        else:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=args.local_train_batch_size,
                                      num_workers=args.num_workers)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.local_eval_batch_size,
                                     num_workers=args.num_workers, pin_memory=True)
        # Define Optimizer and Scheduler.
        optimizer = AdamW([p for n, p in model.named_parameters()], lr=args.learning_rate, eps=1e-8)
        # TODO there may be a bug here.
        # optimizer = AdamW([p for n, p in model.named_parameters() if p.required_grad], lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(len(train_dataloader) * args.epochs * 0.05),
            num_training_steps=len(train_dataloader) * args.epochs
        )
        optimizers = getattr(model, 'optimizers', [optimizer])
        scheduler = getattr(model, 'scheduler', scheduler)

        #model, optimizers = amp.initialize(model.cuda(), optimizers, opt_level="O1")

        trainer = Trainer(log_dir=args.log_dir, model=model, optimizers=optimizers, scheduler=scheduler,
                          pretrained_model=getattr(args, 'pretrained_model', None),
                          use_amp=getattr(args, 'use_amp', False),
                          find_unused_parameters=getattr(args, 'find_unused_parameters', False), adapt=getattr(args, 'adapt_load', True))

        trainer.train(train_loader=train_dataloader, eval_loader=eval_dataloader, epochs=args.epochs,
                      eval_step=getattr(args, 'eval_step', 4), save_step=getattr(args, 'save_step', 4),
                      resume=args.resume, use_tqdm=True,
                      max_norm=getattr(args, 'max_norm', None),
                      gradient_accumulate_steps=getattr(args, 'gradient_accumulate_steps', 1),
                      inner_collect_fn=cf.inner_collect_fn, after_collect_fn=cf.after_collect_fn,
                      best_metric_fn=getattr(model, 'best_metric_fn', lambda x: x['train']['loss_total']))

    if parsed_args.eval_visu:
        logger.warning("Do eval_visu...")
        if local_rank in [-1, 0]:
            ensure_dirname(os.path.join(args.log_dir, 'eval_visu', getattr(args, 'eval_expName', ''), 'analyze'))
        eval_dataset = BaseDataset(args, split=getattr(args, 'visu_split', 'val'))
        if local_rank == -1:
            eval_sampler = SequentialSampler(eval_dataset)
        elif parsed_args.dist:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False)
        else:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=mpu.get_data_parallel_world_size(),
                rank=mpu.get_data_parallel_rank(),
                shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.local_eval_batch_size,
                                     num_workers=args.num_workers, pin_memory=True)

        trainer = Trainer(log_dir=args.log_dir, model=model, pretrained_model=getattr(args, 'pretrained_model', None),
                          find_unused_parameters=getattr(args, 'find_unused_parameters', False), adapt=getattr(args, 'adapt_load', True))
        trainer.eval(eval_dataloader, inner_collect_fn=cf.inner_collect_fn, after_collect_fn=cf.after_collect_fn,
                     use_tqdm=True)


def add_custom_arguments(parser):
    parser.add_argument('--cf', type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--eval_visu', action='store_true')
    parser.add_argument('--eval_clip', action='store_true')
    parser.add_argument('--eval_matrix', action='store_true')
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--megatron', action='store_true')
    parser.add_argument('--tbs', default=None, type=int)
    parser.add_argument('--ebs', default=None, type=int)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = add_custom_arguments(parser)
    parsed_args = parser.parse_args()
    if parsed_args.dist:
        logger.warning('Distributed Training.')
        main_worker(parsed_args.local_rank, parsed_args)
    elif parsed_args.megatron:
        logger.warning('Megatron Training.')
        args_defaults = {'num_layers': 24, 'hidden_size': 1024, 'num_attention_heads': 16, 'seq_length': 1024,
                         'max_position_embeddings': 1024, 'micro_batch_size': 1,
                         'global_batch_size': getattr(parsed_args, 'tbs', 96), 'lr': 0.00015,
                         'train_iters': 500000, 'lr_decay_iters': 320000, 'lr_decay_style': 'cosine',
                         'lr_warmup_fraction': .01, 'seed': 42, 'data_path': '', 'DDP_impl': 'torch',
                         'tensor_model_parallel_size': 1,
                         'pipeline_model_parallel_size': 1}
        initialize_megatron(extra_args_provider=add_custom_arguments,
                            args_defaults=args_defaults)
        megatron_args = megatron.get_args()
        main_worker(megatron_args.local_rank, megatron_args)
    else:
        logger.warning('Common Training.')
        main_worker(-1, parsed_args)
