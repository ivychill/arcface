from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader, distributed
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import os
import torch
import mxnet as mx
from tqdm import tqdm
from data.sampler import DistRandomIdentitySampler
from log import logger

def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
# def get_train_dataset(imgs_folder):
#     train_transform = trans.Compose([
#         trans.RandomHorizontalFlip(),
#         trans.ToTensor(),
#         trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     ds = ImageFolder(str(imgs_folder), train_transform)
#     class_num = ds[-1][1] + 1
#     return ds, class_num
#
# def get_train_loader(conf):
#     if conf.data_mode in ['ms1m', 'concat']:
#         ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
#         print('ms1m loader generated')
#     if conf.data_mode in ['vgg', 'concat']:
#         vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
#         print('vgg loader generated')
#     if conf.data_mode == 'vgg':
#         ds = vgg_ds
#         class_num = vgg_class_num
#     elif conf.data_mode == 'ms1m':
#         ds = ms1m_ds
#         class_num = ms1m_class_num
#     elif conf.data_mode == 'concat':
#         for i,(url,label) in enumerate(vgg_ds.imgs):
#             vgg_ds.imgs[i] = (url, label + ms1m_class_num)
#         ds = ConcatDataset([ms1m_ds,vgg_ds])
#         class_num = vgg_class_num + ms1m_class_num
#     elif conf.data_mode == 'emore':
#         ds, class_num = get_train_dataset(conf.emore_folder/'imgs')
#     elif conf.data_mode == 'glint':
#         ds, class_num = get_train_dataset(conf.glint_folder/'imgs')
#     train_sampler = distributed.DistributedSampler(ds) #add line
#     loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers, sampler = train_sampler)
#     return loader, class_num

def get_test_dataset(imgs_folder):
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, test_transform)
    # logger.debug('dataset: {}'.format(ds.class_to_idx))
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_test_loader(conf):
    # sh_dynamic=conf.data_path/'test'/'shanghai_cam_dynamic_112_test_1k'
    sh_dynamic=conf.data_path/'test'/'kc_employee_dynamic_112'
    sh_dynamic_ds, sh_dynamic_class_num = get_test_dataset(sh_dynamic)
    query_ds = sh_dynamic_ds
    query_class_num = sh_dynamic_class_num

    # sh_id=conf.data_path/'test'/'shanghai_cam_id_112_test_1k'
    sh_id=conf.data_path/'test'/'kc_employee_id_112'
    sh_id_ds, sh_id_class_num = get_test_dataset(sh_id)
    db_id=conf.data_path/'test'/'10w_112'
    db_id_ds, db_id_class_num = get_test_dataset(db_id)
    for i,(url,label) in enumerate(db_id_ds.imgs):
        db_id_ds.imgs[i] = (url, label + sh_id_class_num)
    gallery_ds = ConcatDataset([sh_id_ds, db_id_ds])
    gallery_class_num = sh_id_class_num + db_id_class_num

    # sh_dynamic=conf.data_path/'test'/'q_shanghai_cam_dynamic_112_test_1k'
    # sh_dynamic_ds, sh_dynamic_class_num = get_test_dataset(sh_dynamic)
    # query_ds = sh_dynamic_ds
    # query_class_num = sh_dynamic_class_num
    #
    # sh_id=conf.data_path/'test'/'g_shanghai_cam_dynamic_112_test_1k'
    # sh_id_ds, sh_id_class_num = get_test_dataset(sh_id)
    # gallery_ds = sh_id_ds
    # gallery_class_num = sh_id_class_num

    loader = {}
    loader['query'] = {}
    loader['query']['dl'] = DataLoader(query_ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    loader['query']['cn'] = query_class_num
    loader['query']['len'] = len(query_ds)
    loader['gallery'] = {}
    loader['gallery']['dl'] = DataLoader(gallery_ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    loader['gallery']['cn'] = gallery_class_num
    loader['gallery']['len'] = len(gallery_ds)

    return loader, query_ds, gallery_ds
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    # remarked by fengchen
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = str(path/name), mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        # label = int(header.label)
        label = int(header.label[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # added by fengchen
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)
        # img.save(label_path/'{}.png'.format(idx))
        # img_path = str(label_path/'{}.png'.format(idx))
        # cv2.imwrite(img_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return len(classes), class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, root, class_to_idx, transform=None):
        self.root = root
        extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        dataset = make_dataset(root, class_to_idx, extensions)
        self.dataset = dataset
        self.transform = transform

    def read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not os.path.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid = self.dataset[index]
        img = self.read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid

def get_train_loader(conf, data_mode, sample_identity=False):
    if data_mode == 'emore':
        root = conf.emore_folder/'imgs'
    elif data_mode == 'glint':
        root = conf.glint_folder/'imgs'
    else:
        logger.fatal('invalide data_mode {}'.format(data_mode))
        exit(1)

    class_num, class_to_idx = find_classes(root)
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(root, class_to_idx, train_transform)
    if sample_identity:
        train_sampler = DistRandomIdentitySampler(dataset.dataset, conf.batch_size, conf.num_instances)
    else:
        train_sampler = distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers, sampler = train_sampler)
    return loader, class_num