from config import get_config
from Learner_tmp import face_learner
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist # add line for dist
import os
import time
from log import *
# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    #### seed ####
    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    random.seed(2020)
    torch.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=100, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-1, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=128, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='emore', type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training') # add line for distributed
    parser.add_argument('--fp16', action='store_true',
                        help='use float16 instead of float32, which will save about 50% memory')
    parser.add_argument('--resume', action='store_true',
                        help='resume from previous model via load_state_dict')
    args = parser.parse_args()

    conf = get_config()

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.argsed = args.local_rank
    conf.local_rank = args.local_rank
    conf.fp16 = args.fp16
    conf.resume = args.resume

    #### log ####
    time_str = time.strftime("%Y%m%d_%H%M", time.localtime())
    log_dir = conf.log_path/time_str
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    set_logger(logger, log_dir)
    logger.debug('start train...')
    logger.debug('local_rank {}'.format(args.local_rank))

    learner = face_learner(conf)
    learner.train(conf, args.epochs)