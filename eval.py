# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

import os
import time
import random
import numpy as np
import torch
from torchvision import transforms as trans
import pdb
from config import get_config
import argparse
from Learner import face_learner
from data.data_pipe import get_val_pair
from log import *


def verify(learner):
    vgg2_fp, vgg2_fp_issame = get_val_pair(conf.emore_folder, 'vgg2_fp')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, vgg2_fp, vgg2_fp_issame, nrof_folds=10, tta=True)
    print('vgg2_fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    agedb_30, agedb_30_issame = get_val_pair(conf.emore_folder, 'agedb_30')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, agedb_30, agedb_30_issame, nrof_folds=10, tta=True)
    print('agedb_30 - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    calfw, calfw_issame = get_val_pair(conf.emore_folder, 'calfw')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, calfw, calfw_issame, nrof_folds=10, tta=True)
    print('calfw - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    cfp_ff, cfp_ff_issame = get_val_pair(conf.emore_folder, 'cfp_ff')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cfp_ff, cfp_ff_issame, nrof_folds=10, tta=True)
    print('cfp_ff - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    cfp_fp, cfp_fp_issame = get_val_pair(conf.emore_folder, 'cfp_fp')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cfp_fp, cfp_fp_issame, nrof_folds=10, tta=True)
    print('cfp_fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    cplfw, cplfw_issame = get_val_pair(conf.emore_folder, 'cplfw')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cplfw, cplfw_issame, nrof_folds=10, tta=True)
    print('cplfw - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    lfw, lfw_issame = get_val_pair(conf.emore_folder, 'lfw')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, lfw, lfw_issame, nrof_folds=10, tta=True)
    print('lfw - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)


    # learner.load_state(conf, 'mobilefacenet.pth', False, True)
    vgg2_fp, vgg2_fp_issame = get_val_pair(conf.emore_folder, 'vgg2_fp')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, vgg2_fp, vgg2_fp_issame, nrof_folds=10, tta=True)
    print('vgg2_fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    agedb_30, agedb_30_issame = get_val_pair(conf.emore_folder, 'agedb_30')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, agedb_30, agedb_30_issame, nrof_folds=10, tta=True)
    print('agedb_30 - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    calfw, calfw_issame = get_val_pair(conf.emore_folder, 'calfw')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, calfw, calfw_issame, nrof_folds=10, tta=True)
    print('calfw - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    cfp_ff, cfp_ff_issame = get_val_pair(conf.emore_folder, 'cfp_ff')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cfp_ff, cfp_ff_issame, nrof_folds=10, tta=True)
    print('cfp_ff - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    cfp_fp, cfp_fp_issame = get_val_pair(conf.emore_folder, 'cfp_fp')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cfp_fp, cfp_fp_issame, nrof_folds=10, tta=True)
    print('cfp_fp - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    cplfw, cplfw_issame = get_val_pair(conf.emore_folder, 'cplfw')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cplfw, cplfw_issame, nrof_folds=10, tta=True)
    print('cplfw - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)
    lfw, lfw_issame = get_val_pair(conf.emore_folder, 'lfw')
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, lfw, lfw_issame, nrof_folds=10, tta=True)
    print('lfw - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
    trans.ToPILImage()(roc_curve_tensor)

def identify(learner, conf, log_dir, feat_path, tta):
    learner.compute_true_false_miss(conf, log_dir, feat_path, tta)

if __name__ == '__main__':
    #### seed ####
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    conf = get_config(training=False)

    #### log ####
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = conf.log_path/time_str
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    set_logger(logger, log_dir)
    logger.debug('start eval...')

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    learner = face_learner(conf, inference=True)
    learner.load_state(conf, 'ir_se50.pth', model_only=True, from_save_folder=True)
    # verify(learner)

    # conf.use_mobilfacenet = True
    # learner = face_learner(conf, inference=True)
    # learner.load_state(conf, 'mobilefacenet.pth', True, True)
    # verify(learner)

    conf.gen_feature = True
    feat_path = conf.mat_path/'feature.mat'
    identify(learner, conf, log_dir, feat_path, args.tta)
