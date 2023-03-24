# -*- coding:utf-8 -*-
import collections
import functools
import pandas as pd
import gzip
import itertools
import json
import io
import jsonpickle
import json_lines
import hashlib
import operator
import os
import tempfile
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
import shlex
from datetime import timedelta
from timeit import default_timer
# if pycharm says matplotlib support failed,
# it is because the next deepdish import,
# just try "pip install matplotlib==2.1.2"
import deepdish
import h5py
import nltk
import numpy as np
import time
import scipy.misc
from tqdm import tqdm
import yagmail
import time
import argparse
import copy
import munch
import platform
import imghdr
from PIL import Image
import lmdb
from itertools import groupby
import pickle
from datetime import datetime
from configobj import ConfigObj
from passlib.hash import sha512_crypt
import yaml
import torch
import glob
from collections import defaultdict, OrderedDict
import random
import redis
import msgpack
import importlib
import base64
from functools import partial
import paramiko
import getpass
from stat import S_ISDIR
import posixpath
import requests
from bs4 import BeautifulSoup
import csv
import psutil
import socket
from contextlib import closing
import logging
import colorlog
# from torchsummary import summary
from pprint import pformat
from PIL import Image, ImageSequence
import PIL
import six
import lmdb
import pyarrow as pa
import pathlib
import imageio
########################################################### torch part######################################
# from transformers import BertTokenizer, BertConfig, BertPreTrainedModel
# from transformers.modeling_bert import BertEmbeddings, ACT2FN, gelu, BertIntermediate, BertOutput, BertPooler
# from transformers.modeling_bert import ACT2FN
from transformers import BertTokenizer
from transformers import BertPreTrainedModel

import logging
from torch.nn import CrossEntropyLoss, SmoothL1Loss
import os
import math
import torch
import torch.nn as nn
import math
import torch
# from torch.optim import Optimizer, Adam
from torch.optim import Adam
# from torch.optim.optimizer import required
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils.rnn import pad_sequence

try:
    from megatron import mpu
except Exception as e:
    print('No megatron found.')
# from SimpleTokenizer import SimpleTokenizer
from clip.simple_tokenizer import SimpleTokenizer

#from apex import amp

# Disable transformers outputs weights.
logging.getLogger().setLevel(logging.WARNING)
BertLayerNorm = torch.nn.LayerNorm

########################################################### torch part######################################
# Fix for _csv.Error: field larger than field limit
maxInt = sys.maxsize
decrement = True
while decrement:
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt / 10)
        decrement = True

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def get_logger(filename=None):
    '''
    examples:
    logger = get_logger('try_logging.txt')

    logger.debug("Do something.")
    logger.info("Start print log.")
    logger.warning("Something maybe fail.")
    try:
        raise ValueError()
    except ValueError:
        logger.error("Error", exc_info=True)

    tips:
    DO NOT logger.inf(some big tensors since color may not helpful.)

    '''

    logger = logging.getLogger('utils')
    level = logging.DEBUG
    logger.setLevel(level=level)
    # Use propagate to avoid multiple loggings.
    logger.propagate = False
    # Remove %(levelname)s since we have colorlog to represent levelname.
    format_str = '[%(asctime)s <%(filename)s:%(lineno)d> %(funcName)s] %(message)s'

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    coloredFormatter = colorlog.ColoredFormatter(
        '%(log_color)s' + format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            # 'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'reg,bg_white',
        }
    )

    streamHandler.setFormatter(coloredFormatter)
    logger.addHandler(streamHandler)

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(level)
        formatter = logging.Formatter(format_str)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Fix multiple logging for torch.distributed
    try:
        class UniqueLogger:
            def __init__(self, logger):
                self.logger = logger
                self.local_rank = torch.distributed.get_rank()

            def info(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.info(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.warning(msg, *args, **kwargs)

        logger = UniqueLogger(logger)
    # AssertionError for gpu with no distributed
    # AttributeError for no gpu.
    except Exception:
        pass
    return logger


logger = get_logger()


class DataLoaderX(DataLoader):
    def __iter__(self):
        # transforms generator into a background-thead generator.
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)


def path_join(path, *paths):
    output = os.path.join(path, *paths).replace('\\', '/')
    return output


def str2bool(v):
    if v is None:
        return False
    elif isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Timer:
    def __init__(self):
        '''
        t = Timer()
        time.sleep(1)
        print(t.elapse())
        '''
        self.start = default_timer()

    def elapse(self, readable=False):
        seconds = default_timer() - self.start
        if readable:
            seconds = str(timedelta(seconds=seconds))
        return seconds


def timing(f):
    # 计时器
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        logger.info('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def identity(x):
    return x


def get_parameters(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def adaptively_load_state_dict(target, state_dict, adapt=True):

    if adapt:
        target_dict = target.state_dict()
        # common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

        if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
                target.state_dict()['param_groups'][0]['params']:
            logger.warning('Detected mismatch params, auto adapte state_dict to current')
            common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
        target_dict.update(common_dict)
        target.load_state_dict(target_dict)
        missing_keys = [k for k in target_dict.keys() if k not in common_dict]
        unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

        if len(unexpected_keys) != 0:
            logger.warning(
                f"Some weights of state_dict were not used in target: {unexpected_keys}"
            )
        if len(missing_keys) != 0:
            logger.warning(
                f"Some weights of state_dict are missing used in target {missing_keys}"
            )
        if len(unexpected_keys) == 0 and len(missing_keys) == 0:
            logger.warning("Strictly Loaded state_dict.")
    else:
        target.load_state_dict(state_dict)


def dataset2memory(dataset, use_tqdm=False, num_workers=0, topk=None):
    class MemoryDataset(dataset.__class__):
        def __init__(self):
            # copy all attributes from the father instance
            for k, v in dataset.__dict__.items():
                setattr(self, k, v)
            logger.info('Loading %s into memory, total %s samples.' % (dataset, len(dataset)))
            iter = tqdm if use_tqdm else identity
            dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=identity, batch_size=1)

            self.data = []
            for i, e in enumerate(iter(dataloader)):
                if i >= len(dataset) or (topk and i >= topk):
                    break
                # self.data.append(e[0])
                self.data.append(copy.deepcopy(e[0]))

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    return MemoryDataset()


def dataset2lmdb(dataset, output_dirname, processor=None, map_size=20 * (1024 ** 3), write_frequency=5000,
                 num_workers=0, batch_size=1, topk=np.Infinity):
    logger.info(
        'Converting Dataset (%s samples, batch_size %s) into %s.' % (len(dataset), batch_size, output_dirname))
    # if os.path.exists(output_dirname):
    #     raise ValueError('Existing %s, please remove it manually.' % output_dirname)
    data_loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=lambda x: x)
    db = lmdb.open(output_dirname, subdir=True,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)

    time = Timer()
    txn = db.begin(write=True)
    keys = []
    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if idx >= topk:
            break
        if processor:
            batch = processor(batch)
        for sample in batch:
            sample_id = sample[0]
            value = sample[1]
            txn.put(sample_id.encode('utf-8'), pa.serialize(value).to_buffer())
            keys.append(sample_id)
            if idx % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pa.serialize(keys).to_buffer())
    db.sync()
    db.close()
    logger.warning("Successfully write %s samples to %s, used %s."
                   % (len(keys), output_dirname, time.elapse(readable=True)))


def groupby(l, key=lambda x: x):
    d = collections.defaultdict(list)
    for item in l:
        d[key(item)].append(item)
    return dict(d.items())


def list_filenames(dirname, filter_fn=None, sort_fn=None, printable=True):
    dirname = os.path.abspath(dirname)
    filenames = os.listdir(dirname)
    filenames = [os.path.join(dirname, filename) for filename in filenames]
    if filter_fn:
        tmp = len(filenames)
        if printable:
            logger.info('Start filtering files in %s by %s.' % (dirname, filter_fn))
        filenames = [e for e in filenames if filter_fn(e)]
        if printable: logger.info(
            'Detected %s files/dirs in %s, filtering to %s files.' % (tmp, dirname, len(filenames)))
    else:
        if printable: logger.info('Detected %s files/dirs in %s, No filtering.' % (len(filenames), dirname))
    if sort_fn:
        filenames = sorted(filenames, key=sort_fn)

    return filenames


def listdict2dict2list(listdict, printable=True):
    tmp_dict = collections.defaultdict(list)
    for example_dict in listdict:
        for k, v in example_dict.items():
            tmp_dict[k].append(v)
    if printable: logger.info('%s' % tmp_dict.keys())
    return dict(tmp_dict)


class Meter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if isinstance(val, (int, float)):
            self.val = val
            if self.sum:
                self.sum += val * n
            else:
                self.sum = val * n
            if self.count:
                self.count += n
            else:
                self.count = n
            self.avg = self.sum / self.count
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    val[k] = v.item()
            if self.val:
                for k in val.keys():
                    self.val[k] = val[k]
            else:
                self.val = val
            if self.sum:
                for k in val.keys():
                    if k in self.sum:
                        self.sum[k] = self.sum[k] + val[k] * n
                    else:
                        self.sum[k] = val[k] * n
            else:
                self.sum = {k: val[k] * n for k in val.keys()}
            if self.count:
                for k in val.keys():
                    if k in self.count:
                        self.count[k] = self.count[k] + n
                    else:
                        self.count[k] = n
            else:
                self.count = {k: n for k in val.keys()}
            self.avg = {k: self.sum[k] / self.count[k] for k in self.count.keys()}
        else:
            raise ValueError('Not supported type %s' % type(val))

    def __str__(self):
        if isinstance(self.avg, dict):
            return str({k: "%.4f" % v for k, v in self.avg.items()})


def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname


def add_suffix(filename, suffix):
    dirname, rootname, extname = split_filename(filename)
    output_filename = os.path.join(dirname, "%s%s.%s" % (rootname, suffix, extname))
    return output_filename


def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        elif extname == 'msg':
            with open(filename, 'wb') as f:
                msgpack.dump(data, f)
        elif extname == 'h5':
            if kwargs is None:
                params = {}
            split_num = kwargs.get('split_num')

            if split_num:
                if not isinstance(data, list):
                    raise ValueError(
                        '[error] utils.data2file: data must have type of list when use split_num, but got %s' % (
                            type(data)))

                if not split_num <= len(data):
                    raise ValueError(
                        '[error] utils.data2file: split_num(%s) must <= data(%s)' % (len(split_num), len(data)))

                print_save_flag = False
                print_did_not_save_flag = False
                pre_define_filenames = ["%s_%d" % (filename, i) for i in range(split_num)]
                pre_search_filenames = glob.glob("%s*" % filename)

                strict_existed = (set(pre_define_filenames) == set(pre_search_filenames) and len(
                    set([os.path.exists(e) for e in pre_define_filenames])) == 1)
                common_existed = len(set([os.path.exists(e) for e in pre_search_filenames])) == 1

                def rewrite():
                    logger.info('Spliting data to %s parts before saving...' % split_num)
                    data_splits = np.array_split(data, indices_or_sections=split_num)
                    for i, e in enumerate(data_splits):
                        deepdish.io.save("%s_%d" % (filename, i), list(e))
                    logger.info('Saved data to %s_(0~%d)' % (
                        os.path.abspath(filename), len(data_splits) - 1))

                if strict_existed and not override:
                    logger.info(
                        'Did not save data to %s_(0~%d) because the files strictly exist and override is False' % (
                            os.path.abspath(filename), len(pre_search_filenames) - 1))
                elif common_existed:
                    logger.warning('Old wrong files (maybe a differnt split) exist, auto delete them.')
                    for e in pre_search_filenames:
                        os.remove(e)
                    rewrite()
                else:
                    rewrite()
            else:
                deepdish.io.save(filename, data)
        elif extname == 'hy':
            # hy support 2 params: key and max_step
            # if key, then create group using key, else create group using index
            # if max_step, then the loop may early stopping, used for debug
            # Remove filename since h5py may corrupt.
            if override:
                remove_filename(filename)
            key_str = kwargs.pop('key_str', None)
            topk = kwargs.pop('topk', None)

            with h5py.File(filename, 'w') as f:
                for i, datum in enumerate(tqdm(data)):
                    if key_str:
                        grp = f.create_group(name=datum[key_str])
                    else:
                        grp = f.create_group(name=str(i))
                    for k in datum.keys():
                        grp[k] = datum[k]
                    if topk is not None and i + 1 == topk:
                        break
        elif extname == 'csv':
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif extname == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f)
        elif extname == 'npy':
            np.save(filename, data)
        elif extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            imageio.mimsave(filename, data, format='GIF', duration=kwargs.get('duration'))
        # elif extname == 'ckpt':
        #     tf.train.Saver().save(data, filename)
        # elif extname == 'jpg' or extname == 'png':
        #     plt.imsave(filename, data)
        elif extname == 'pth':
            torch.save(data, filename)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('type can only support h5, csv, json, sess')
        if printable: logger.info('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: logger.info(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    if extname == 'pkl':
        with open(filename, 'rb') as f:
            # data = pickle.load(f, encoding='latin1')
            data = pickle.load(f)
    elif extname == 'msg':
        with open(filename, 'rb') as f:
            data = msgpack.load(f, encoding="utf-8")
    elif extname == 'h5':
        split_num = kwargs.get('split_num')
        if split_num:
            print_load_flag = False
            if isinstance(split_num, int):
                filenames = ["%s_%i" % (filename, i) for i in range(split_num)]
                if split_num != len(glob.glob("%s*" % filename)):
                    logger.warning('Maybe you are giving a wrong split_num(%d) != seached num (%d)' % (
                        split_num, len(glob.glob("%s*" % filename))))

            elif split_num == 'auto':
                filenames = glob.glob("%s*" % filename)
                logger.info('Auto located %d splits linked to %s' % (len(filenames), filename))
            else:
                raise ValueError("params['split_num'] got unexpected value: %s, which is not supported." % split_num)
            data = []
            for e in filenames:
                data.extend(deepdish.io.load(e))
            logger.info('Loaded data from %s_(%s)' % (
                os.path.abspath(filename), ','.join(sorted([e.split('_')[-1] for e in filenames]))))
        else:
            data = deepdish.io.load(filename)
    elif extname == 'csv':
        data = pd.read_csv(filename)
    elif extname == 'tsv':  # Returns generator since tsv file is large.
        if not kwargs.get('delimiter'):  # Set default delimiter
            kwargs['delimiter'] = '\t'
        if not kwargs.get('fieldnames'):  # Check field names
            raise ValueError('You must specify fieldnames when load tsv data.')
        # Required args.
        key_str = kwargs.pop('key_str')
        decode_fn = kwargs.pop('decode_fn')
        # Optimal args.
        topk = kwargs.pop('topk', None)
        redis = kwargs.pop('redis', None)
        if not redis:
            data = dict()
        else:
            data = redis
        if not redis or not redis.check():
            with open(filename) as f:
                reader = csv.DictReader(f, **kwargs)
                for i, item in enumerate(tqdm(reader)):
                    if not redis:  # if memory way
                        decode_fn(item)
                    data[item[key_str]] = item
                    if topk is not None and i + 1 == topk:
                        break
        else:
            logger.warning('check_str %s in redis, skip loading.' % data.check_str)
    elif extname == 'hy':
        data = h5py.File(filename, 'r')
        # print('[info] utils.file2data: size: %d, keys: %s' % (len(f.keys()), list(f['0'].keys())))
    elif extname in ['npy', 'npz']:
        try:
            data = np.load(filename, allow_pickle=True)
        except UnicodeError:
            logger.warning('%s is python2 format, auto use latin1 encoding.' % os.path.abspath(filename))
            data = np.load(filename, encoding='latin1', allow_pickle=True)
    elif extname == 'json':
        with open(filename) as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError('[error] utils.file2data: failed to load json file %s' % filename)
    elif extname == 'jsonl':
        with open(filename, 'rb') as f:
            data = [e for e in json_lines.reader(f)]
    elif extname == 'ini':
        data = ConfigObj(filename, encoding='utf-8')
    elif extname == 'pth':
        data = torch.load(filename, map_location=kwargs.get('map_location'))

        # try:
        #     data = torch.load(filename)
        # except RuntimeError as e:
        #     logger.warning('Auto map location to cpu.')
        #     data = torch.load(filename, map_location=torch.device('cpu'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            logger.info('Loaded data from %s' % os.path.abspath(filename))
    return data


def download_file(fileurl, filedir=None, progress_bar=True, override=False, fast=False, printable=True):
    if filedir:
        ensure_dirname(filedir)
        assert os.path.isdir(filedir)
    else:
        filedir = ''
    filename = os.path.abspath(os.path.join(filedir, fileurl.split('/')[-1]))
    # print(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.info("%s not exist, automatic makedir." % dirname)
    if not os.path.exists(filename) or override:
        if fast:
            p = subprocess.Popen('axel -n 10 -o {0} {1}'.format(filename, fileurl), shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(p.stdout.readline, ''):
                if line:
                    logger.info(line.decode('utf-8').replace('\n', ''))
                else:
                    p.kill()
                    break
        else:
            if progress_bar:
                def my_hook(t):
                    last_b = [0]

                    def inner(b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            t.total = tsize
                        t.update((b - last_b[0]) * bsize)
                        last_b[0] = b

                    return inner

                with tqdm(unit='B', unit_scale=True, miniters=1,
                          desc=fileurl.split('/')[-1]) as t:
                    urllib.request.urlretrieve(fileurl, filename=filename,
                                               reporthook=my_hook(t), data=None)
            else:
                urllib.request.urlretrieve(fileurl, filename=filename)
        if printable: logger.info("%s downloaded sucessfully." % filename)
    else:
        if printable: logger.info("%s already existed" % filename)
    return filename


def extract_file(filename, targetname="HERE", override=False, printable=True):
    assert os.path.exists(filename)
    dirname, rootname, extname = split_filename(filename)

    if targetname == 'HERE':
        targetname = os.path.abspath(dirname)
    elif targetname == 'NEW':
        targetname = os.path.join(dirname, rootname)
    else:
        targetname = os.path.abspath(targetname)

    if targetname == os.path.abspath(dirname) or override or not os.path.exists(targetname):
        if extname == 'tar' or extname == 'tar.gz':
            with tarfile.open(filename) as f:
                for e in f.getnames():
                    f.extract(e, path=targetname)
        elif extname == 'zip':
            with zipfile.ZipFile(filename) as f:
                f.extractall(path=targetname)
        elif extname == 'gz':
            with gzip.GzipFile(filename) as f, open(os.path.join(targetname, rootname), "wb") as t:
                t.write(f.read())
        else:
            raise ValueError("Only support tar, tar.gz, zip, gz")
        if printable: logger.info("Extracted sucessfully to %s " % targetname)
    else:
        if printable: logger.info("%s already existed" % targetname)


def copy_file(filename, targetname, override=False, printable=True):
    filename = os.path.abspath(filename)
    targetname = os.path.abspath(targetname)
    if not os.path.exists(targetname) or override:
        shutil.copy2(filename, targetname)
        # with open(filename, 'r') as f1, open(targetname, 'w') as f2:
        #     shutil.copyfileobj(f1, f2)
        if printable:
            logger.info('Copied %s to %s.' % (filename, targetname))
    else:
        if printable:
            logger.info('Did not copy because %s exists.' % targetname)


def compress_file(filename, targetname=None, type='zip', override=False, printable=True):
    if targetname is None:
        targetname = os.path.abspath("%s.%s" % (filename, type))
    filename = os.path.abspath(filename)
    if not os.path.exists(targetname) or override:
        if type == 'zip':
            with zipfile.ZipFile(targetname, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(filename, arcname=os.path.basename(filename))
            if printable:
                logger.info('Compressed %s to %s.' % (filename, targetname))
        else:
            raise ValueError('Only support type zip now, but got %s' % type)
    else:
        if printable:
            logger.info('Did not compress because %s exists.' % targetname)
    return targetname


def clean_path(path):
    while os.path.exists(path):
        shutil.rmtree(path)
    while not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        logger.info('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        logger.info('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)

        # if override:
        #     shutil.rmtree(dirname)
        # if not os.path.exists(dirname):
        #     print('[info] utils.ensure_dirname: making dirname: %s' % os.path.abspath(dirname))
        #     os.makedirs(dirname)


def ensure_filename(filename, override=False):
    dirname, rootname, extname = split_filename(filename)
    ensure_dirname(dirname, override=False)
    if os.path.exists(filename) and override:
        os.remove(filename)
        logger.info('Deleted filename %s' % filename)


def remove_filename(filename, printable=False):
    if os.path.isfile(filename) or os.path.islink(filename):
        os.remove(filename)
        if printable:
            logger.info('Deleted file %s.' % filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)
        if printable:
            logger.info('Deleted dir %s.' % filename)
    else:
        raise ValueError("%s is not a file or dir." % filename)



def sentencelist2wordlist(sentencelist):
    return list(itertools.chain(*[e.split() for e in sentencelist]))


def flattenlist(nestedlist):
    return list(itertools.chain(*nestedlist))


def length2sublist(length, num_sublist):
    spacing = np.linspace(0, length, num_sublist + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    return ranges


def split_length(length, step=None, num=None):
    if step:
        assert not num
        assert step <= length
    else:
        assert num
        assert num <= length

    assert (not step and num) or (not num and step)
    if num:
        step = int(np.ceil(length / num))

    spacing = list(np.arange(0, length, step)) + [length]
    if num and len(spacing) - 1 < num:
        x = length - num
        spacing = spacing[0:x] + [i for i in range(spacing[x], length + 1)]

    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append(list(range(spacing[i], spacing[i + 1])))
    return ranges


def sentence2wordlist(sentence, start=None, end=None):
    s = sentence.split()
    tmp = []
    if start:
        tmp.append(start)
    tmp.extend(s)
    if end:
        tmp.append(end)
    return tmp


def tokenizerV1(s):
    t_str = s.lower()
    for i in ['?', '!', '\'', '\"', '$', ':', r'@', r'(', r')', ',', '.', ';']:
        t_str = t_str.replace(i, '')
    for i in ['-', '/']:
        t_str = t_str.replace(i, ' ')
    q_list = [e for e in t_str.split(' ') if e]
    return q_list


def tokenizerV2(s):
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
    return s


def xnor(x, y):
    if (x and y) or (not x and not y):
        return True
    else:
        return False


def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    logger.info('\n'.join(table))


def expand_list(l, length, fill=0, direction='right'):
    tmp = [fill] * length
    if direction == 'left':
        tmp[0:len(l)] = l[0: length]
        return tmp
    elif direction == 'right':
        tmp[-len(l):] = l[0: length]
        return tmp
    else:
        raise ValueError("diretion can only be left or right")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def mul(l):
    return functools.reduce(operator.mul, l)


class Vocabulary(object):
    def __init__(self, original_wordlist, special_wordlist=None, vocabulary_size=None, min_word_count=None,
                 init_dict=None, name='', printable=True):

        self._special_wordlist = special_wordlist
        self.vocabulary_size = vocabulary_size
        self.min_word_count = min_word_count
        self.name = name
        self.printable = printable

        self.vocabulary_wordlist = None
        self.emb_array = None
        self._word2idx = None
        self._idx2word = None
        self._build_vocabulary(original_wordlist, init_dict)

    def _build_vocabulary(self, original_wordlist, init_dict):
        if self.printable:
            logger.info("==Start building vocabulary %s ==" % self.name)

        counter = collections.Counter(original_wordlist)
        sorted_count = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if self._special_wordlist:
            tmp = []
            for k, v in sorted_count:
                if k not in self._special_wordlist:
                    tmp.append((k, v))
                else:
                    if self.printable:
                        logger.info("Special_wordlist element %s is in original_wordlist" % k)
            sorted_count = tmp
        self._try_size = []
        for tmpi in range(1, 100):
            sorted_count_part = [e for e in sorted_count if e[1] >= tmpi]
            self._try_size.append(
                (tmpi, len(sorted_count_part),
                 "%.3f%%" % (sum([e[1] for e in sorted_count_part]) / len(original_wordlist) * 100)))
        if self.printable:
            tmp = "[info] utils.build_vocabulary: trying_min_word_count1~5: %s" % self._try_size[0:5]
            if self._special_wordlist:
                tmp += " + %d" % len(self._special_wordlist)
            logger.info(tmp)

        if self.min_word_count:
            self.vocabulary_size = len([e for e in sorted_count if e[1] >= self.min_word_count])
            if self._special_wordlist:
                self.vocabulary_size += len(self._special_wordlist)
        elif not self.vocabulary_size:
            self.vocabulary_size = len(sorted_count)
            if self._special_wordlist:
                self.vocabulary_size += len(self._special_wordlist)

        sorted_final = [k for k, v in sorted_count]
        if self._special_wordlist:
            sorted_final = self._special_wordlist + sorted_final

        sorted_final = sorted_final[:self.vocabulary_size]
        self._word2idx = {e: i for i, e in enumerate(sorted_final)}
        self._idx2word = {v: k for k, v in self._word2idx.items()}
        self.vocabulary_wordlist = sorted_final

        if init_dict:
            emb_size = next(iter(init_dict.values())).shape[0]
            self.emb_array = np.zeros((self.vocabulary_size, emb_size), dtype=np.float32)
            for idx, word in self._idx2word.items():
                if word in init_dict:
                    self.emb_array[idx] = init_dict[word]

        if self.printable:
            logger.info("original_wordlist_size: %d" % len(original_wordlist))
            logger.info("original_wordset_size: %d" % len(set(original_wordlist)))
            logger.info("target vocabulary_size: %d" % self.vocabulary_size)
            if init_dict:
                logger.info("emb_size: %d" % emb_size)
            logger.info("Ended building vocabulary %s ==" % self.name)

    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            if self._special_wordlist:
                return 1
            else:
                return -1
                # raise ValueError(
                #     '[error] utils.Vocabulary.word2idx: Word %s is not in Vocabulary %s and you did not set UNK' % (
                #         word, self.name))

    def idx2word(self, idx):
        if idx in self._idx2word:
            return self._idx2word[idx]
        else:
            raise ValueError('[error] tools.Dictionary: %s is not a vocab index.' % idx)

    def wordlist2idxlist(self, wordlist):
        return [self.word2idx(e) for e in wordlist]

    def idxlist2wordlist(self, idxlist):
        return [self.idx2word(e) for e in idxlist]

    def wordlist2sparselist(self, wordlist):
        sparselist = np.zeros([self.vocabulary_size], dtype=np.uint8)
        for e in wordlist:
            sparselist[self.word2idx(e)] = 1
        return sparselist

    def get_idlist_length(self, idlist):
        assert self.special_wordlist
        return idlist.index(0)

    # def get_original_idxlist(self, one_hot=False):
    #     idxlist = [self.word2idx(e) for e in self._original_wordlist]
    #     if one_hot:
    #         idxlist = np.array([np.identity(self.vocabulary_size)[idx] for idx in idxlist])
    #     return idxlist

    def __len__(self):
        return self.vocabulary_size


def func_name():
    return sys._getframe(1).f_code.co_name


def class_name(self):
    return self.__class__.__name__


def update_python(fileordirname=None):
    if not fileordirname:
        fileordirname = os.getcwd()
    os.system('2to3 {0} -w'.format(fileordirname))


def execute(cmd, wait=True, printable=True):
    if wait:
        if printable: logger.warning('Executing: '"%s"', waiting...' % cmd)
        # if platform.system() == 'Windows':
        #     cmd = cmd.replace(r'\', '\\')
        try:
            output = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            logger.warning(e.output.decode('utf-8'))
            output = None
            # sys.exit(-1)

        return output
    else:
        if platform.system() == 'Windows':
            black_hole = 'NUL'
        elif platform.system() == 'Linux':
            black_hole = '/dev/null'
        else:
            raise ValueError('Unsupported system %s' % platform.system())
        cmd = cmd + ' 1>%s 2>&1' % black_hole
        if printable: logger.info('Executing: '"%s"', not wait.' % cmd)
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)



def merge_tuple_list(tuple_list, fn=None):
    tuple_list = sorted(tuple_list, key=lambda x: x[0], reverse=True)
    if fn:
        return [(key, fn([num for _, num in value])) for key, value in
                itertools.groupby(tuple_list, lambda x: x[0])]
    else:
        return [(key, [num for _, num in value]) for key, value in
                itertools.groupby(tuple_list, lambda x: x[0])]


# format

def format_img(fileordirname, type='jpg'):
    fileordirname = os.path.abspath(fileordirname)

    def format_a_img(filename):
        dirname, rootname, extname = split_filename(filename)
        if extname in ['jpg', 'png', 'jpeg']:
            rawtype = imghdr.what(filename)
            if rawtype == 'jpeg':
                rawtype = 'jpg'
            if rawtype != type:
                dirname, rootname, extname = split_filename(filename)
                img = Image.open(filename)
                img.save(os.path.join(dirname, rootname + '.' + type))
                logger.info('Formatting image file %s from %s to %s' % (
                    os.path.abspath(filename), rawtype, type))

    if os.path.isfile(fileordirname):
        format_a_img(fileordirname)
    elif os.path.isdir(fileordirname):
        for parent, dirnames, filenames in tqdm(os.walk(fileordirname)):
            for filename in filenames:
                target = os.path.join(parent, filename)
                format_a_img(target)


def check_socket(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # dirname = os.path.dirname(filename)
    # if dirname not in sys.path:
    #     sys.path.append(dirname)
    return module


def get_permutation(height, width):
    """ Get the permutation corresponding to a snake-like walk as decribed by the paper. Used to flatten the convolutional feats. """
    permutation = np.zeros(height * width, np.int32)
    for i in range(height):
        for j in range(width):
            permutation[i * width + j] = i * width + j if i % 2 == 0 else (i + 1) * width - j - 1
    return permutation


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_image(image_filename):
    img = open(image_filename, 'rb')
    try:
        if Image.open(img).convert('RGB') is None:
            logger.warning('%s is a broken image.' % image_filename)
            img.close()
            return False
        img.close()
        return True
    except Exception as e:
        logger.warning('%s is a broken image.' % image_filename)
        img.close()
        return False


def get_stem(path):
    return pathlib.Path(path).stem


class SimpleFolder(Dataset):

    def __init__(self, folder_dirname, id_fn=get_stem, filter_fn=None):
        super(SimpleFolder, self).__init__()
        self.id_fn = id_fn
        self.filenames = list_filenames(folder_dirname, filter_fn=filter_fn, printable=True)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_bytes = self.raw_reader(filename)
        sample_id = self.id_fn(filename)
        return sample_id, data_bytes

    def __len__(self):
        return len(self.filenames)

    def raw_reader(self, path):
        with open(path, 'rb') as f:
            bin_data = f.read()
        return bin_data


class LMDBFolder(Dataset):
    def __init__(self, lmdb_dirname):
        self.lmdb_dirname = lmdb_dirname
        self.env = lmdb.open(lmdb_dirname, subdir=True, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))
        self.key2idx = {e.decode(): i for i, e in enumerate(self.keys)}

    def __getitem__(self, index):
        sample_id = self.keys[index].decode('utf-8')
        with self.env.begin(write=False) as txn:
            data_bytes = pa.deserialize(txn.get(self.keys[index]))
        return sample_id, data_bytes

    def get(self, sample_id):
        index = self.key2idx[sample_id]
        date_bytes = self.__getitem__(index)[1]
        return date_bytes

    def __len__(self):
        return self.length


def pname2pid(str_proc_name):
    map_proc_info = {}
    for proc in psutil.process_iter():
        if proc.name() == str_proc_name:
            map_proc_info[proc.pid] = str_proc_name

    return map_proc_info


class LMDBDict:
    def __init__(self, lmdb_dir='lmdb_dir', map_size=20 * (1024 ** 3)):
        self.lmdb_dir = lmdb_dir
        self.env = lmdb.open(lmdb_dir, map_size=map_size)
        self.txn = self.env.begin(write=False)
        self.keys = pa.deserialize(self.txn.get(b'__keys__'))

    def __getitem__(self, key):
        return pa.deserialize(self.txn.get(str(key).encode()))

    def __len__(self):
        return len(self.keys)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()


def file2hash(filename):
    # md5 = hashlib.md5()
    # with open(filename, 'rb') as f:
    #     data = f.read()
    # md5.update(data)
    # return md5.hexdigest()
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class UniqueFile():
    def __init__(self, dirname):
        self.dirname = dirname
        # 初始化config, {'....jpg': None}
        self.config_filename = os.path.join(dirname, 'config.json')
        if os.path.exists(self.config_filename):
            self.config = file2data(self.config_filename)
        else:
            self.config = dict()

    def save(self, filename, delete_duplicate=False):
        rootname_unique = file2hash(filename)
        dirname, rootname, extname = split_filename(filename)
        basename_unique = '%s.%s' % (rootname_unique, extname)
        filename_unique = os.path.join(self.dirname, basename_unique)
        if basename_unique in self.config:
            if delete_duplicate:
                os.remove(filename)
                logger.info('Deleted %s because it already exists in %s.' % (filename, filename_unique))
            else:
                logger.info('Noting done because %s already exists in %s.' % (filename, filename_unique))

        else:
            self.update(basename_unique, None)
            # self.config[basename_unique] = None
            # data2file(self.config, self.config_filename, override=True)
            if delete_duplicate:
                shutil.move(filename, filename_unique)
                logger.info('Successfully moved %s to %s' % (filename, filename_unique))
            else:
                shutil.copy2(filename, filename_unique)
                logger.info('Successfully copied %s to %s' % (filename, filename_unique))
        return filename_unique

    def update(self, k, v):
        self.config[k] = v
        data2file(self.config, self.config_filename, override=True)


class UniqueFileAuto():
    def __init__(self, dirname):
        self.dirname = dirname

    def save(self, filename, delete_duplicate=False):
        rootname_unique = file2hash(filename)
        dirname, rootname, extname = split_filename(filename)
        basename_unique = '%s.%s' % (rootname_unique, extname)
        filename_unique = os.path.join(self.dirname, basename_unique)
        if basename_unique in os.listdir(self.dirname):
            if delete_duplicate:
                os.remove(filename)
                logger.info('Deleted %s because it already exists in %s.' % (filename, filename_unique))
            else:
                logger.info('Noting done because %s already exists in %s.' % (filename, filename_unique))

        else:
            if delete_duplicate:
                shutil.move(filename, filename_unique)
                logger.info('Successfully moved %s to %s' % (filename, filename_unique))
            else:
                shutil.copy2(filename, filename_unique)
                logger.info('Successfully copied %s to %s' % (filename, filename_unique))
        return filename_unique


class Transmit:
    def __init__(self, hostname, port, username, password, private_key=None, auto_skip=False):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.private_key = private_key
        self.auto_skip = auto_skip
        self.ssh = None
        self.sftp = None
        self.is_connected = False

    def connect(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        logger.info('Connection %s started.' % self.hostname)
        try:
            if self.private_key:
                self.ssh.connect(self.hostname, self.port, self.username, pkey=
                paramiko.RSAKey.from_private_key_file(self.private_key, password=self.password))
            else:
                self.ssh.connect(self.hostname, self.port, self.username, self.password, timeout=2)
            self.sftp = self.ssh.open_sftp()
            self.is_connected = True
            logger.info('Connection %s established.' % self.hostname)
        except Exception as e:
            self.is_connected = False
            if self.auto_skip:
                logger.warning('Connection %s failed.' % self.hostname)
            else:
                raise e
        return self

    def put(self, local_filename, remote_filename, override=False):
        # You must input absolute file names.
        if not self.exists(remote_filename):
            self.makedirs(os.path.dirname(remote_filename), exist_ok=True)
            logger.info('Auto mkdir: %s' % os.path.dirname(remote_filename))
        if not self.exists(remote_filename) or override:
            self.sftp.put(local_filename, remote_filename)
            logger.info('From %s to %s@%s:%s' % (local_filename, self.username, self.hostname, remote_filename))
        else:
            logger.info('Nothing done because %s@%s:%s exists and override is set False' %
                        (self.username, self.hostname, remote_filename))

    def get(self, remote_filename, local_filename, override=False, printable=True):
        # You must input absolute file names.
        if not os.path.exists(local_filename):
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
            if printable: logger.info('Auto mkdir: %s' % os.path.dirname(local_filename))
        if not os.path.exists(local_filename) or override:
            self.sftp.get(remote_filename, local_filename)
            if printable: logger.info(
                'From %s@%s:%s to %s' % (self.username, self.hostname, remote_filename, local_filename))
        else:
            if printable: logger.info(
                'Nothing done because %s exists and override is set False' % local_filename)

    def list_filenames(self, dirname, filter_fn=None, sort_fn=None, printable=True):
        filenames = self.sftp.listdir(dirname)
        if filter_fn:
            L = len(filenames)
            filenames = [e for e in filenames if filter_fn(e)]
            if printable: logger.info(
                'Detected %s files/dirs in %s, filtering to %s files.' % (
                    L, dirname, len(filenames)))
        else:
            if printable: logger.info(
                'Detected %s files/dirs in %s, No filtering.' % (
                    len(filenames), dirname))
        if sort_fn:
            filenames = sorted(filenames, key=sort_fn)
        filenames = [path_join(dirname, filename) for filename in filenames]
        return filenames

    def _walk_remote(self, dirpath):
        dirnames = []
        filenames = []

        for fd in self.sftp.listdir_attr(dirpath):
            if S_ISDIR(fd.st_mode):
                dirnames.append(fd.filename)
            else:
                filenames.append(fd.filename)
        yield dirpath, dirnames, filenames

        for dirname in dirnames:
            new_dirpath = os.path.join(dirpath, dirname)
            for walk in self._walk_remote(new_dirpath):
                yield walk

    def get_dir(self, remote, local, override=False, printable=True):
        logger.info(local)
        if not os.path.exists(local):
            os.makedirs(local, exist_ok=True)
            if printable: logger.info('Auto mkdir: %s' % os.path.dirname(local))
        st_mode = self.sftp.stat(remote).st_mode
        if not S_ISDIR(st_mode):
            # remote is not directory
            remote_dir = os.path.basename(remote)
            if not os.path.exists(local) or override:
                self.sftp.get(remote, local)
                if printable: logger.info(
                    'From %s@%s:%s to %s' % (self.username, self.hostname, remote, local))
            else:
                if printable: logger.info(
                    '<Tools.get>: warning: nothing done because %s exists and override is set False' % local)
        else:
            logger.info('Recurively transmit data from remote directory %s to local directory %s' % (
                remote, local))
            parent, child = os.path.split(remote)
            logger.info(parent, child)
            for dirpath, dirnames, filenames in self._walk_remote(remote):
                logger.info(dirpath, dirnames, filenames)
                # dirpath = dirpath.replace(parent, '.')
                for dirname in dirnames:
                    try:
                        os.makedirs(os.path.join(local, dirpath, dirname))
                    except:
                        pass
                for filename in filenames:
                    localpath = os.path.join(local, dirpath, filename)
                    remotepath = os.path.join(parent, dirpath, filename)
                    logger.info(remotepath, localpath)
                    self.sftp.get(remotepath, localpath)

    def node(self):
        _, stdout, _ = self.ssh.exec_command("python -c 'import platform;print(platform.node())'")
        n = stdout.read().decode().strip()
        if n:
            return n
        else:
            raise SystemError('You need to pip install platform first.')

    def exists(self, path):
        try:
            self.sftp.stat(path)
            return True
        except Exception:
            return False

    def abspath(self, path):
        # This will auto check whether the path exists.
        return self.sftp.normalize(path)

    def isdir(self, path):
        try:
            return S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            # Path does not exist, so by definition not a directory
            return False

    def makedirs(self, name, mode=0o777, exist_ok=False):
        head, tail = posixpath.split(name)
        if not tail:
            head, tail = posixpath.split(head)
        if head and tail and not self.exists(head):
            try:
                self.makedirs(head, mode, exist_ok)
            except FileExistsError:
                # Defeats race condition when another thread created the path
                pass
            cdir = '.'
            if isinstance(tail, bytes):
                cdir = bytes('.', 'ASCII')
            if tail == cdir:  # xxx/newdir/. exists if xxx/newdir exists
                return
        try:
            self.sftp.mkdir(name, mode)
        except OSError:
            # Cannot rely on checking for EEXIST, since the operating system
            # could give priority to other errors like EACCES or EROFS
            if not exist_ok or not self.isdir(name):
                raise OSError('%s already exists.' % name)


class TransmitPhilly:
    def __init__(self, auto_skip=False):
        self.auto_skip = auto_skip
        self.is_connected = False
        pass

    def connect(self):
        url = 'https://storage.eu1.philly.selfhost.corp.microsoft.com/nextmsra/t-chwu/'
        if self.auto_skip:
            try:
                self.is_connected = requests.get(url)
            except requests.exceptions.ConnectionError:
                logger.exception('warning! cannot connect.')
        else:
            self.is_connected = requests.get(url)
        return self

    def list_filenames(self, dirname, filter_fn=None, sort_fn=None, printable=True):
        bs = BeautifulSoup(requests.get(dirname).text, 'lxml')
        filenames = [e.get('href') for e in bs.select('#list > tbody a') if e.get('href') != '../']
        if filter_fn:
            tmp = len(filenames)

            filenames = [e for e in filenames if filter_fn(e)]
            if printable: logger.info(
                'Detected %s files/dirs in %s, filtering to %s files.' % (tmp, dirname, len(filenames)))
        else:
            if printable: logger.info('Detected %s files/dirs in %s, No filtering.' % (len(filenames), dirname))
        if sort_fn:
            filenames = sorted(filenames, key=sort_fn)
        filenames = [path_join(dirname, filename) for filename in filenames]
        return filenames

    def exists(self, filename):
        return requests.get(filename).ok

    def get(self, remote_filename, local_filename, override=False, printable=True):
        local_dirname = os.path.dirname(local_filename)
        if not os.path.exists(local_filename) or override:
            ensure_dirname(local_dirname)
            download_file(remote_filename, filedir=local_dirname, printable=printable)
        else:
            if printable: logger.info('Nothing done because %s exists and override is set False' % local_filename)

    def download_filename(self, root_dir, filename, override=False):
        local_filedir = path_join(root_dir, '/'.join(filename.split('/')[-4:-1]))
        local_filename = path_join(root_dir, '/'.join(filename.split('/')[-4:]))
        if not os.path.exists(local_filename) or override:
            ensure_dirname(local_filedir)
            download_file(filename, filedir=local_filedir)
        else:
            logger.info('%s already exists.' % local_filename)

    def download_method(self, root_dir, l, v, method_name='MutanVarOn', part_fn=None):
        epoch_dirnames = self.list_filenames(
            'https://storage.%s.philly.selfhost.corp.microsoft.com/%s/t-chwu/data/GQA/logs/%s/' % (
                l, v, method_name))

        for epoch_dirname in epoch_dirnames:
            remote_filenames = self.list_filenames(epoch_dirname)
            if part_fn:
                remote_filenames = part_fn(remote_filenames)
            for remote_filename in remote_filenames:
                self.download_filename(root_dir, remote_filename)

def iterable_class(cls):
    def iterfn(self):
        iters = dict((x, y) for x, y in cls.__dict__.items() if x[:2] != '__')
        iters.update(self.__dict__)

        for x, y in iters.items():
            yield x, y

    cls.__iter__ = iterfn
    return cls


def PIL2bytes(image: Image):
    imgByteArr = io.BytesIO()
    if not image.format:
        image.format = 'PNG'
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


'''
Pytorch part:
'''


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, seed=None, p=None, af=None, dim=None):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)
        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class InverseSqrtLR(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 lr: float,
                 warmup_init_lr: float,
                 warmup_steps: int,
                 last_epoch: int = -1,
                 num_updates: int = 0) -> None:

        warmup_end_lr = lr
        if warmup_init_lr < 0:
            warmup_init_lr = warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_steps
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_steps ** 0.5
        # initial learning rate
        self.lr = warmup_init_lr
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.optimizer = optimizer
        self.set_lr()
        self.num_updates = num_updates
        super().__init__(optimizer, last_epoch=last_epoch)

    def step(self):
        if self.num_updates < self.warmup_steps:
            self.lr = self.warmup_init_lr + self.num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * self.num_updates ** -0.5
        self.set_lr()

    def set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_lr(self):
        return self.lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.set_lr()


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Trainer:
    '''
        There is still bugs when resume on windows.
    '''

    def reduce_mean(self, tensor):
        rt = tensor.clone()
        size = int(os.environ['WORLD_SIZE'])
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt = rt / size
        return rt

    def wrap_model(self):
        if hasattr(self.model, 'module'):
            raise ValueError('You do not need to wrap a model with modules.')

        if self.mode == 'common':
            logger.info('Wrapped model to common %s.' % self.device)

            self.model.to(self.device)
            if self.n_gpu > 1:
                logger.warning('Detected %s gpus, auto using DataParallel.' % self.n_gpu)
                self.model = torch.nn.DataParallel(self.model)
        elif self.mode == 'dist':
            logger.info('Wrapped model to distributed %s.' % self.device)

            self.device = torch.device("cuda", self.local_rank)
            self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=self.find_unused_parameters)
        elif self.mode == 'megatron':
            self.model.cuda()
            devid = torch.cuda.current_device()
            # enable bucketing if no tensor parallelism is used
            bucket_cap_mb = 25 if mpu.get_tensor_model_parallel_world_size() == 1 else 1024 * 40
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[devid], output_device=devid,
                process_group=self.hybrid_info['ddp_parallel_group'],
                bucket_cap_mb=bucket_cap_mb, find_unused_parameters=self.find_unused_parameters
            )
        else:
            raise ValueError
        # wrap_optimizers
        if self.optimizers:
            for i in range(len(self.optimizers)):
                self.optimizers[i].load_state_dict(
                    complex_to_device(self.optimizers[i].state_dict(), device=self.device))

    def __init__(self, log_dir, model, optimizers=None, scheduler=None, pretrained_model=None, use_amp=True,
                 find_unused_parameters=True, adapt=True):
        # Basic Params
        self.log_dir = log_dir
        self.model = model
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.pretrained_model = pretrained_model
        self.use_amp = use_amp
        self.find_unused_parameters = find_unused_parameters
        # Load Pretrained Models.
        if pretrained_model:
            self.from_pretrained(pretrained_model, adapt)
        # Get Variables from ENV
        self.rank = int(os.getenv('RANK', '-1'))
        self.local_rank = int(os.getenv('LOCAL_RANK', '-1'))
        # Define Running mode.
        if self.local_rank == -1:
            self.mode = 'common'
            self.enable_write_model = True
            self.enable_collect = True
            self.enable_write_metric = True
        elif mpu.is_unitialized():
            self.mode = 'dist'
            self.enable_write_model = (self.rank == 0)
            self.enable_collect = True
            self.enable_write_metric = (self.rank == 0)
        else:
            self.mode = 'megatron'
            self.enable_write_model = (mpu.get_data_parallel_rank() == 0)
            self.enable_collect = (mpu.get_tensor_model_parallel_rank() == 0)
            self.enable_write_metric = (self.rank == 0)
            self.hybrid_info = dict(
                tensor_model_parallel_src_rank=mpu.get_tensor_model_parallel_src_rank(),
                tensor_model_parallel_group=mpu.get_tensor_model_parallel_group(),
                ddp_parallel_rank=mpu.get_data_parallel_rank(),
                ddp_parallel_group=mpu.get_data_parallel_group())

        if self.enable_write_metric:
            ensure_dirname(log_dir, override=False)
        self.metric_filename = os.path.join(log_dir, 'metric.json')
        if self.mode == 'megatron':
            # TODO make sure each rank can load the model of last epoch
            self.checkpoint_filename = os.path.join(log_dir, 'last-chunk-{}.pth'.format(
                mpu.get_tensor_model_parallel_rank()))
            self.best_checkpoint_filename = os.path.join(log_dir, 'best-chunk-{}.pth'.format(
                mpu.get_tensor_model_parallel_rank()))
            self.each_checkpoint_filename = os.path.join(log_dir, 'epoch%s-chunk-{}.pth'.format(
                mpu.get_tensor_model_parallel_rank()))
        else:
            self.checkpoint_filename = os.path.join(log_dir, 'last.pth')
            self.best_checkpoint_filename = os.path.join(log_dir, 'best.pth')
            self.each_checkpoint_filename = os.path.join(log_dir, 'epoch%s.pth')
        self.collect_filename = os.path.join(log_dir, 'collect.pth')
        self.epoch = -1

        # Get device and number of GPUs
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu >= 1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.use_amp and self.n_gpu < 1:
            raise ValueError('AMP Does not support CPU!')
        if self.use_amp and self.mode == 'common':
            logger.warning('In common mode, remember to @autocast before forward function.')
        self.scalar = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def check_outputs(self, outputs):
        error_message = 'Model output must be a dict. The key must be "class_subclass" format.' \
                        ' "class" can only be loss, metric, or logits. "subclass" should be a string.' \
                        ' But got an unexpected key %s'
        loss_total_list = [e for e in outputs.keys() if e.startswith('loss_total')]
        if not loss_total_list:
            raise ValueError('Model output must contain a key startswith "loss_total"!')

        for k, v in outputs.items():
            split_res = k.split('_')
            if len(split_res) < 2:
                raise ValueError(error_message % k)
            if k.split('_')[0] not in ['loss', 'metric', 'logits']:
                raise ValueError(error_message % k)

    def train(self, train_loader, eval_loader=None, epochs=5, resume=True, eval_step=10,
              save_step=None, use_tqdm=None,
              max_norm=None, gradient_accumulate_steps=1, inner_collect_fn=None, after_collect_fn=None,
              best_metric_fn=lambda x: x['train']['loss_total']):
        if not save_step:
            save_step = eval_step
        best_eval_metric = np.Infinity
        if resume:
            if os.path.exists(self.checkpoint_filename):
                self.load_checkpoint(self.checkpoint_filename)
        else:
            if self.enable_write_metric:
                logger.warning('Dangerous! You set resume=False. Auto cleaning all the logs under %s' % self.log_dir)
                ensure_dirname(self.log_dir, override=True)
        self.wrap_model()
        if self.enable_write_metric:
            ensure_dirname(self.log_dir, override=False)
        epoch_iter = range(self.epoch + 1, epochs, 1)
        if len(epoch_iter):
            logger.warning('Start train & val phase...')
        else:
            logger.warning('Skip train & val phase...')
        # Train & Eval phase
        logger.warning('Train examples: %s, local_batch_size: %s, epochs: %s.' % (
            len(train_loader.dataset), train_loader.batch_size, epochs))
        for epoch in epoch_iter:
            self.epoch = epoch
            # Train phase
            train_meter, train_time = self.train_fn(train_loader,
                                                    max_norm=max_norm,
                                                    gradient_accumulate_steps=gradient_accumulate_steps,
                                                    use_tqdm=use_tqdm)
            logger.info('[Rank %s] Train Epoch: %d/%d, Time: %s\n %s' %
                        (self.rank, epoch + 1, epochs, train_time, train_meter.avg))
            if not isinstance(train_meter.avg, dict):
                raise ValueError(type(train_meter.avg))
            metric = {'Epoch%s' % (epoch + 1): {'train': {**train_meter.avg, **{'time': train_time}}}}

            if self.enable_write_metric:
                self.update_metric_file(metric)
            if (epoch + 1) % save_step == 0:
                if self.enable_write_model:
                    self.save_checkpoint(self.checkpoint_filename)
                    copy_file(self.checkpoint_filename, self.each_checkpoint_filename % str(epoch + 1), override=True)

            if (epoch + 1) % eval_step == 0:
                if eval_loader:
                    # print('debug.....')
                    eval_meter, eval_time, collect_dict = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                                       after_collect_fn=after_collect_fn,
                                                                       use_tqdm=use_tqdm)

                    logger.info('[Rank %s] Valid Epoch: %d/%d, Time: %s\n %s' %
                                (self.rank, epoch + 1, epochs, eval_time, eval_meter.avg))

                    # Update metric with eval metrics
                    metric['Epoch%s' % (epoch + 1)].update({'eval': {**eval_meter.avg, **{'time': eval_time}}})

                    # Save metric file
                    if self.enable_write_metric:
                        self.update_metric_file(metric)

                    # If the best model, save another checkpoint.
                    if best_metric_fn(metric['Epoch%s' % (epoch + 1)]) < best_eval_metric and self.enable_write_model:
                        best_eval_metric = best_metric_fn(metric['Epoch%s' % (epoch + 1)])
                        if os.path.exists(self.checkpoint_filename):
                            copy_file(self.checkpoint_filename, self.best_checkpoint_filename, override=True)
                        else:
                            logger.warning('No checkpoint_file %s' % self.checkpoint_filename)

    def eval(self, eval_loader, inner_collect_fn=None, after_collect_fn=None, use_tqdm=True):
        # This function is used to do evaluating after training.
        if not self.pretrained_model:
            raise ValueError('You must create a new config file and specify pretrained_model in Args when using eval.')
        # Wrap model before evaluating. This will support ddp evaluating.
        self.wrap_model()
        eval_meter, eval_time, _ = self.eval_fn(eval_loader, inner_collect_fn=inner_collect_fn,
                                                after_collect_fn=after_collect_fn, use_tqdm=use_tqdm)
        logger.info('[Rank %s] Valid Time: %s\n %s' % (self.rank, eval_time, eval_meter.avg))

    def predict(self, inputs, device):
        # This function is used to predict a single sample.
        # device = [v.device for k, v in inputs.items() if getattr(v, 'device', False)][0]
        self.model = self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            inputs = complex_to_device(inputs, device)
            outputs = self.model(inputs)
        return outputs

    def update_metric_file(self, metric):
        if os.path.exists(self.metric_filename):
            r = file2data(self.metric_filename, printable=False)
            data2file(dict(r, **metric), self.metric_filename, override=True)
        else:
            data2file(metric, self.metric_filename)

    def train_fn(self, train_loader, max_norm, gradient_accumulate_steps=1, use_tqdm=True):
        self.model.train()
        train_meter = Meter()
        train_timer = Timer()
        train_iter = tqdm(train_loader, total=len(train_loader), disable=not use_tqdm)
        for step, inputs in enumerate(train_iter):
            for optimizer_idx in range(len(self.optimizers)):
                if not getattr(self.optimizers[optimizer_idx], 'is_enabled', lambda x: True)(self.epoch):
                    continue
                # TODO add non_blocking=True
                inputs = complex_to_device(inputs, self.device)
                if self.mode == 'megatron':
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            torch.distributed.broadcast(
                                v,
                                self.hybrid_info['tensor_model_parallel_src_rank'],
                                self.hybrid_info['tensor_model_parallel_group']
                            )
                            torch.cuda.synchronize()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    inputs['epoch'] = self.epoch
                    inputs['global_step'] = self.epoch * len(train_loader) + step
                    inputs['optimizer_idx'] = optimizer_idx
                    outputs = self.model(inputs)
                    self.check_outputs(outputs)
                    # If we use nn.Parallel, we will get a list of metric or losses from different GPUs, we need to mean them.
                    if self.mode == 'common' and self.n_gpu > 1:
                        for k, v in outputs.items():
                            if k.split('_')[0] in ['metric', 'loss']:
                                outputs[k] = v.mean()

                    # If we use gradient_accumulation, we need to divide loss by gradient_accumulate_steps
                    if gradient_accumulate_steps != 1:
                        # for k, v in outputs.items():
                        #     if k.split('_')[0] in ['loss']:
                        #         outputs[k] = v / gradient_accumulate_steps
                        for k in outputs.keys():
                            if k.startswith('loss_total'):
                                outputs[k] = outputs[k] / gradient_accumulate_steps
                if (step + 1) % gradient_accumulate_steps == 0:
                    self.optimizers[optimizer_idx].zero_grad()

                if optimizer_idx == 0:
                    #with amp.scale_loss(outputs['loss_total'], self.optimizers[optimizer_idx]) as scaled_loss:
                    #    scaled_loss.backward()
                    self.scalar.scale(outputs['loss_total']).backward()
                else:
                    #with amp.scale_loss(outputs['loss_total_%s' % optimizer_idx], self.optimizers[optimizer_idx]) as scaled_loss:
                    #    scaled_loss.backward()
                    self.scalar.scale(outputs['loss_total_%s' % optimizer_idx]).backward()
                if (step + 1) % gradient_accumulate_steps == 0:
                    if max_norm:
                        self.scalar.unscale_(self.optimizers[optimizer_idx])
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    self.scalar.step(self.optimizers[optimizer_idx])
                    self.scalar.update()
                    if self.scheduler:
                        self.scheduler.step()
                    # self.model.zero_grad()
                    # self.optimizers[optimizer_idx].zero_grad()
                metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                if self.mode != 'common':
                    for k, v in metric_and_loss.items():
                        metric_and_loss[k] = self.reduce_mean(v)
                train_meter.update(metric_and_loss)

            train_iter.set_description("Metering:" + str(train_meter))
        train_time = train_timer.elapse(True)
        return train_meter, train_time

    def eval_fn(self, eval_loader, inner_collect_fn=None, after_collect_fn=None, use_tqdm=True):
        # TODO Note that eval_fn supports ddp, megatron. So we do not need to unwrap things here.
        model_to_eval = self.model
        model_to_eval.eval()
        eval_meter = Meter()
        eval_timer = Timer()
        collect_list = []
        with torch.no_grad():
            eval_loader = tqdm(eval_loader, total=len(eval_loader)) if use_tqdm else eval_loader
            for inputs in eval_loader:
                # TODO add non_blocking=True
                inputs = complex_to_device(inputs, self.device)
                if self.mode == 'megatron':
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            torch.distributed.broadcast(
                                v,
                                self.hybrid_info['tensor_model_parallel_src_rank'],
                                self.hybrid_info['tensor_model_parallel_group']
                            )
                            torch.cuda.synchronize()
                outputs = model_to_eval(inputs)
                self.check_outputs(outputs)

                # If we use nn.Parallel, we will get a list of metric or losses from different GPUs, we need to mean them.
                if self.mode == 'common' and self.n_gpu > 1:
                    for k, v in outputs.items():
                        if k.split('_')[0] in ['metric', 'loss']:
                            outputs[k] = v.mean()
                metric_and_loss = {k: v for k, v in outputs.items() if k.split('_')[0] in ['metric', 'loss']}
                if self.mode != 'common':
                    for k, v in metric_and_loss.items():
                        metric_and_loss[k] = self.reduce_mean(v)
                eval_meter.update(metric_and_loss)

                # workaround to only use one GPU for update
                if inner_collect_fn and self.enable_collect:
                    collect_list.extend(inner_collect_fn(inputs, outputs, model_to_eval))
            # If set after_collect_fn, collect_list will be changed by this function.
            if after_collect_fn and self.enable_collect:
                collect_list = after_collect_fn(collect_list, eval_meter, self.log_dir, self.epoch)
        eval_time = eval_timer.elapse(True)
        return eval_meter, eval_time, collect_list

    def load_checkpoint(self, checkpoint_filename):
        if hasattr(self.model, "module"):
            raise ValueError("Please do not load checkpoint into wrapped model, ensure self.model is CPU.")
        checkpoint = file2data(checkpoint_filename, map_location='cpu')
        # self.model.load_state_dict(checkpoint['model'], strict=False)
        adaptively_load_state_dict(self.model, checkpoint['model'])
        # self.from_pretrained(checkpoint)
        if self.optimizers:
            if len(self.optimizers) > 1:
                for i, optimizer in enumerate(self.optimizers):
                    # self.optimizers[i].load_state_dict(checkpoint['optimizer'][i], strict=False)
                    adaptively_load_state_dict(self.optimizers[i], checkpoint['optimizer'][i])

            elif len(self.optimizers) == 1:
                # self.optimizers[0].load_state_dict(checkpoint['optimizer'], strict=False)
                adaptively_load_state_dict(self.optimizers[0], checkpoint['optimizer'])

            else:
                raise ValueError
        if self.scheduler:
            # self.scheduler.load_state_dict(checkpoint['scheduler'], strict=False)
            # self.scheduler.load_state_dict(checkpoint['scheduler'], strict=False)
            adaptively_load_state_dict(self.scheduler, checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] - 1

        # IMPORTANT! The model will be wrapped automatically.
        logger.warning('Loaded checkpoint %s of epoch %s' % (checkpoint_filename, checkpoint['epoch']))

    def save_checkpoint(self, checkpoint_filename):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if len(self.optimizers) > 1:
            optimizer_to_save = [optimizer.state_dict() for optimizer in self.optimizers]
        elif len(self.optimizers) == 1:
            optimizer_to_save = self.optimizers[0].state_dict()
        else:
            raise ValueError
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': optimizer_to_save,
            'epoch': self.epoch + 1,
        }
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        data2file(checkpoint, checkpoint_filename, override=True)
        logger.warning('Saved epoch %s to %s.' % (checkpoint['epoch'], checkpoint_filename))

    def from_pretrained(self, pretrained_model, adapt=True):
        if hasattr(self.model, "module"):
            raise ValueError("Please do not load pretrained model into wrapped model, ensure self.model is CPU.")
        if isinstance(pretrained_model, str):
            logger.warning('Loading Pretrained Model Path: %s...' % pretrained_model)
            pretrained_dict = file2data(pretrained_model, map_location='cpu')['model']
        else:
            logger.warning('Loading Given Pretrained Dict...')
            pretrained_dict = pretrained_model
        adaptively_load_state_dict(self.model, pretrained_dict, adapt)




def dl2ld(dl):
    return [dict(zip(dl, e)) for e in zip(*dl.values())]


def ld2dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def default_inner_collect_fn(inputs, outputs, _):
    dl = {**inputs, **{k: v for k, v in outputs.items() if k.split('_')[0] == 'logits'}}
    ld = dl2ld(dl)
    return ld


def calculate_acc(inputs, target):
    _, pred = torch.max(inputs, 1)  # pred: b, [3, 4]
    if target.dim() == 1:
        return torch.mean((pred == target).float()) * 100

    elif target.dim() == 2:
        return torch.gather(target, dim=1, index=pred.unsqueeze(1)).sum() * 100 / pred.size(0)


def calculate_seq_acc(inputs, target, mask):
    _, predicted = torch.max(inputs, -1)  # b*L
    return (((predicted == target).float() * mask).sum(1) / mask.sum(1)).mean() * 100


def calculate_seq_iou(inputs, target, mask):
    _, predicted = torch.max(inputs, -1)  # b*L
    i = (predicted * target * mask).sum(-1).float()
    u = (((predicted + target) * mask) > 0).sum(-1).float()
    return (i / u).mean() * 100


def complex_to_device(complex, device, non_blocking=False):
    if isinstance(complex, torch.Tensor):
        return complex.to(device, non_blocking=non_blocking)
    elif isinstance(complex, dict):
        return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
    elif isinstance(complex, list) or isinstance(complex, tuple):
        return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
    elif isinstance(complex, str) or isinstance(complex, bytes) or \
            isinstance(complex, int) or isinstance(complex, float):
        return complex
    else:
        raise ValueError('Unsupported complex', complex)


def dict_collate_fn(batch, handle_dict={}):
    from torch._six import string_classes, int_classes, container_abcs
    _use_shared_memory = False
    numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }

    def default_collate(batch, handle_fn=None):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        if handle_fn:
            return handle_fn(batch)

        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], int_classes):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], container_abcs.Mapping):
            try:
                return {key: default_collate([d[key] for d in batch], handle_fn=handle_dict.get(key)) for key in
                        batch[0]}
            except Exception as e:
                print("debugger")
        elif isinstance(batch[0], container_abcs.Sequence):
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    return default_collate(batch, handle_fn=None)


class CLIPSimilarity(nn.Module):
    import clip
    def __init__(self, model_filename):
        super(CLIPSimilarity, self).__init__()
        self.model_filename = model_filename
        self.model, self.p = self.clip.load(model_filename, device='cpu')
        self.model = self.model.eval()

    def forward(self, text, image, batch_size=None):
        '''
        text: [X]
          for example,  ['str_1', 'str_2', ..., 'str_X']
        image: [Y, c, w, h]

        return: [X, Y]
        '''
        device = image.device
        # print(self.p.transforms)
        # import ipdb
        # ipdb.set_trace()
        # img_input = F.interpolate(image, size=self.p.transforms[0].size[0])
        img_input = F.interpolate(image, size=224)
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        img_input = (img_input.clamp(min=-1.0, max=1.0) + 1.0) / 2.0
        img_input -= image_mean[:, None, None]
        img_input /= image_std[:, None, None]
        text_input = self.clip.tokenize(text, truncate=True).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input).float()
            image_features = self.model.encode_image(img_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logit_scale = 1.0  # Default value is model.logit_scale.exp()
            # logits_per_image = logit_scale * image_features @ text_features.t()
            if batch_size:
                text_features = rearrange(text_features, '(b x) d -> b x d', b=batch_size)
                image_features = rearrange(image_features, '(b y) d -> b d y', b=batch_size)
            else:
                image_features = image_features.t()
            logits_per_text = logit_scale * torch.matmul(text_features, image_features)
        similarity = logits_per_text
        return similarity


def execute_cmd(cmd, input_data=None):
    process = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate(input=input_data)
    retcode = process.poll()
    if retcode:
        raise ValueError(err.decode('utf-8'))
    return out


def videofile2videobytes(input_video):
    out = execute_cmd('ffmpeg -y -i %s -c copy -movflags +faststart -f nut pipe:' % input_video, input_data=None)
    return out


def videofile2videometa(input_video):
    out = execute_cmd('ffprobe -i %s -print_format json -show_streams' % input_video)
    meta = json.loads(out.decode('utf-8'))

    if 'duration' in meta['streams'][0]:
        duration = float(meta['streams'][0]['duration'])
    else:  # Fix Duration for webm format.
        duration_str = meta['streams'][0]['tags']['DURATION']
        h, m, s = duration_str.split(':')
        duration = float(h) * 3600 + float(m) * 60 + float(s)

    res = {'width': meta['streams'][0]['width'],
           'height': meta['streams'][0]['height'],
           'duration': duration,
           'fps': eval(meta['streams'][0]['r_frame_rate'])}
    return res


def videobytes2videofile(input_bytes, output_video):
    execute_cmd('ffmpeg -y -i pipe: %s' % output_video, input_data=input_bytes)


def videobytes2videoarr(input_bytes, seek_start=None, seek_duration=None, seek_fps=None, wh=None):
    if wh:
        width, height = wh
    else:
        ffprob_out = execute_cmd('ffprobe -i pipe: -print_format json -show_streams', input_data=input_bytes)
        meta = json.loads(ffprob_out.decode('utf-8'))
        width = meta['streams'][0]['width']
        height = meta['streams'][0]['height']
    cmd = 'ffmpeg -y -i pipe: '
    if seek_start:
        cmd += f'-ss {seek_start} '
    if seek_duration:
        cmd += f'-t {seek_duration} '
    if seek_fps:
        cmd += f'-filter_complex [0]fps=fps={seek_fps}[s0] -map [s0] '
    cmd += '-f rawvideo -pix_fmt rgb24 pipe:'
    #     assert cmd == 'ffmpeg -y -i pipe: -ss 2 -t 4 -filter_complex [0]fps=fps=0.5[s0] -map [s0] -f rawvideo -pix_fmt rgb24 pipe:'
    ffmpeg_out = execute_cmd(cmd, input_data=input_bytes)
    video = np.frombuffer(ffmpeg_out, np.uint8)
    video = video.reshape([-1, height, width, 3])
    return video


def videofile2videoarr(input_file, seek_start=None, seek_duration=None, seek_fps=None):
    ffprob_out = execute_cmd(f'ffprobe -i {input_file} -print_format json -show_streams')
    meta = json.loads(ffprob_out.decode('utf-8'))
    width = meta['streams'][0]['width']
    height = meta['streams'][0]['height']
    cmd = f'ffmpeg -y -i {input_file} '
    if seek_start:
        cmd += f'-ss {seek_start} '
    if seek_duration:
        cmd += f'-t {seek_duration} '
    if seek_fps:
        cmd += f'-filter_complex [0]fps=fps={seek_fps}[s0] -map [s0] '
    cmd += '-f rawvideo -pix_fmt rgb24 pipe:'
    #     assert cmd == 'ffmpeg -y -i pipe: -ss 2 -t 4 -filter_complex [0]fps=fps=0.5[s0] -map [s0] -f rawvideo -pix_fmt rgb24 pipe:'
    ffmpeg_out = execute_cmd(cmd)
    video = np.frombuffer(ffmpeg_out, np.uint8)
    video = video.reshape([-1, height, width, 3])
    return video

def gif_to_duration(gif_filename):
    """ Returns the average framerate of a PIL Image object """
    PIL_Image_object = Image.open(gif_filename)
    PIL_Image_object.seek(0)
    frames = duration = 0
    while True:
        try:
            frames += 1
            duration += PIL_Image_object.info['duration']
            PIL_Image_object.seek(PIL_Image_object.tell() + 1)
        except EOFError:
            return duration
    return None


def gif_to_frames(gif_filename, output_format='tensor'):
    if output_format not in ['tensor', 'array', 'PIL']:
        raise ValueError('Not supported output_format %s.' % output_format)
    im = Image.open(gif_filename)
    PIL_frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
    if output_format == 'PIL':
        return PIL_frames
    array_frames = np.array(
        [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in PIL_frames])
    if output_format == 'array':
        return array_frames
    tensor_frames = [transforms.ToTensor()(frame) for frame in array_frames]
    if output_format == 'tensor':
        return tensor_frames

if __name__ == '__main__':
    pass
    # sf = Stanford()
    # t = sf.question_to_parsertree(["What color is the man's shirt"]*10000)
    # ts = t.split('\n\n')
    # print("Done!")
