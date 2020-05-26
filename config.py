from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    # conf.data_path = Path('../dataset/')
    conf.data_path = Path('/opt/nfs/192/')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.mat_path = conf.work_path/'mat'
    conf.input_size = [112,112]
    conf.embedding_size = 256
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    # conf.drop_ratio = 0.6
    conf.drop_ratio = 0.4
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    # conf.data_mode = 'glint'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'train'/'faces_emore'
    conf.glint_folder = conf.data_path/'train'/'faces_glint'
    conf.gallery_folder = conf.data_path/'train'/'faces_emore_plus_sh_32w'
    conf.argsed = None
    conf.batch_size = 100 # irse net depth 50 
#   conf.batch_size = 200 # mobilefacenet
    conf.num_instances = 4
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-1
        conf.milestones = [100000,160000,220000]
        # conf.milestones = [200000, 320000, 440000]
        # conf.milestones = [8,12,16]
        # conf.milestones = [9,14,19]
        # conf.milestones = [6, 10, 14, 18]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()
        conf.fp16 = True

#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
        conf.pin_memory = True
        conf.num_workers = 8
        conf.gen_feature = True
    return conf
