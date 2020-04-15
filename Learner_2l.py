# -*- coding: utf-8 -*-

from data.data_pipe import de_preprocess, get_train_loader, get_val_data, get_test_loader
from model_2l import Backbone, Arcface, MobileFaceNet, Triplet, l2_norm
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras, extract_feature
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
from pathlib import Path
from collections import OrderedDict
from identification import compute_rank1, DataPath
import os
import scipy.io
import xlwt
from log import logger

try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.parallel import DistributedDataParallel
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

class face_learner(object):
    def __init__(self, conf, inference=False):
        accuracy = 0.0
        logger.debug(conf)
        if conf.use_mobilfacenet:
            # self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            self.model = MobileFaceNet(conf.embedding_size).cuda()
            logger.debug('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).cuda()#.to(conf.device)
            logger.debug('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        if not inference:
            self.milestones = conf.milestones
            logger.info('loading data...')
            self.loader_arc, self.class_num_arc = get_train_loader(conf, 'emore', sample_identity=False)
            self.loader_tri, self.class_num_tri = get_train_loader(conf, 'emore', sample_identity=True)

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head_arc = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num_arc).cuda()
            self.head_tri = Triplet().cuda()
            logger.debug('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head_arc.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head_arc.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            # self.optimizer = torch.nn.parallel.DistributedDataParallel(optimizer,device_ids=[conf.argsed])
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            if conf.fp16:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O3")
                self.model = DistributedDataParallel(self.model).cuda()
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[conf.argsed], find_unused_parameters=True).cuda() #add line for distributed

            self.board_loss_every = len(self.loader_arc)//100
            self.evaluate_every = len(self.loader_arc)//2
            self.save_every = len(self.loader_arc)//2
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(Path(self.loader_arc.dataset.root).parent)
        else:
            self.threshold = conf.threshold
            self.loader, self.query_ds, self.gallery_ds = get_test_loader(conf)

    def save_state(self, conf, epoch, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
                                     ('model_{}_{}_acc:{:.4f}_{}.pth'.format(epoch, self.step, accuracy,
                                                                                   extra)))
        if not model_only:
            torch.save(
                self.head_arc.state_dict(), save_path /
                                         ('head_{}_{}_acc:{:.4f}_{}.pth'.format(epoch, self.step, accuracy,
                                                                                       extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                         ('optimizer_{}_{}_acc:{:.4f}_{}.pth'.format(epoch, self.step, accuracy,
                                                                                extra)))

    def load_network(self, save_path):
        state_dict = torch.load(save_path)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # logger.debug('key {}'.format(k))
            namekey = k[7:]
            # logger.debug('key {}'.format(namekey))  # remove 'module.'
            new_state_dict[namekey] = v
        # load params
        return new_state_dict

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        if conf.resume:
            self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str)))
        else:
            self.model.load_state_dict(self.load_network(save_path / 'model_{}'.format(fixed_str)))

        if not model_only:
            self.head_arc.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
            logger.info('load optimizer {}'.format(self.optimizer))
            # amp.load_state_dict(torch.load(save_path / 'amp_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda())[1] + self.model(fliped.cuda())[1]
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.cuda())[1].cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda())[1] + self.model(fliped.cuda())[1]
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.cuda())[1].cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    # true top 1, false top 1, miss
    def compute_true_false_miss(self, conf, log_dir, feat_path, tta):
        def gen_distmat(qf, q_pids, gf, g_pids):
            m, n = qf.shape[0], gf.shape[0]
            logger.debug('query shape {}, gallery shape {}'.format(qf.shape, gf.shape))
            # logger.debug('q_pids {}, g_pids {}'.format(q_pids, g_pids))
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()
            return distmat

        def distance(emb1, emb2):
            diff = np.subtract(emb1, emb2)
            dist = np.sum(np.square(diff), 1)
            return dist

        self.model.eval()
        if conf.gen_feature:
            with torch.no_grad():
                query_feature, query_label = extract_feature(conf, self.model, self.loader['query']['dl'], tta)
                gallery_feature, gallery_label = extract_feature(conf, self.model, self.loader['gallery']['dl'], tta)
            # result = {'query_feature': query_feature.numpy(), 'query_label': query_label,
            #     'gallery_feature': gallery_feature.numpy(), 'gallery_label': gallery_label}

            result = {'query_feature': query_feature.numpy(), 'query_label': query_label.numpy(),
                'gallery_feature': gallery_feature.numpy(), 'gallery_label': gallery_label.numpy()}
            scipy.io.savemat(feat_path, result)

        else:
            result = scipy.io.loadmat(feat_path)
            query_feature = torch.from_numpy(result['query_feature'])
            query_label = torch.from_numpy(result['query_label'])[0]
            gallery_feature = torch.from_numpy(result['gallery_feature'])
            gallery_label = torch.from_numpy(result['gallery_label'])[0]

        distmat = gen_distmat(query_feature, query_label, gallery_feature, gallery_label)

        # record txt
        with open(os.path.join(log_dir, 'result.txt'),'at') as f:
            f.write('%s\t%s\t%s\t%s\n' % ('threshold', 'acc', 'err', 'miss'))

        # record excel
        xls_file = xlwt.Workbook()
        sheet_1 = xls_file.add_sheet('sheet_1', cell_overwrite_ok=True)
        row = 0
        path_excel = os.path.join(log_dir, 'result.xls')

        sheet_title = ['threshold', 'acc', 'err', 'miss']
        for i_sheet in range(len(sheet_title)):
            sheet_1.write(row, i_sheet, sheet_title[i_sheet])
        xls_file.save(path_excel)
        row += 1


        index = np.argsort(distmat)  # from small to large
        max_index = index[:, 0]

        query_list_file = 'data/probe.txt'
        gallery_list_file = 'data/gallery.txt'
        err_rank1 = os.path.join(log_dir, 'err_rank1.txt')
        data_path = DataPath(query_list_file, gallery_list_file)
        with open(err_rank1,'at') as f:
            f.write('%s\t\t\t%s\n' % ('query', 'gallery'))

        thresholds = np.arange(0.4, 2, 0.01)
        for threshold in thresholds:
            acc, err, miss = compute_rank1(distmat, max_index, query_label, gallery_label, threshold, data_path, err_rank1)
            # record txt
            with open(os.path.join(log_dir, 'result.txt'),'at') as f:
                f.write('%.6f\t%.6f\t%.6f\t%.6f\n' % (threshold, acc, err, miss))

            # record excel
            list_data = [threshold, acc, err, miss]
            for i_1 in range(len(list_data)):
                sheet_1.write(row, i_1, list_data[i_1])
            xls_file.save(path_excel)
            row += 1


    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head_arc(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model.train()
        # logger.debug('model {}'.format(self.model))
        running_loss = 0.

        # 断点加载训练
        if conf.resume:
            logger.debug('resume...')
            self.load_state(conf, 'ir_se100.pth', from_save_folder=True)

        logger.debug('optimizer {}'.format(self.optimizer))
        for epoch in range(epochs):
            logger.debug('epoch {} started'.format(epoch))
            for data_arc, data_tri in zip(tqdm(iter(self.loader_arc)), tqdm(iter(self.loader_tri))):
                if self.step in self.milestones:
                    self.schedule_lr()
                imgs_arc, labels_arc = data_arc
                # logger.debug('labels_arc {}'.format(labels_arc))
                imgs_tri, labels_tri = data_tri
                # logger.debug('labels_tri {}'.format(labels_tri))
                imgs_arc = imgs_arc.cuda()
                labels_arc = labels_arc.cuda()
                imgs_tri = imgs_tri.cuda()
                labels_tri = labels_tri.cuda()
                self.optimizer.zero_grad()
                _, embeddings_arc = self.model(imgs_arc)
                embeddings_tri, _ = self.model(imgs_tri)
                thetas = self.head_arc(embeddings_arc, labels_arc)
                loss_arc = conf.ce_loss(thetas, labels_arc)
                loss_tri = self.head_tri(embeddings_tri, labels_tri)
                loss = loss_arc + loss_tri
                if conf.fp16:  # we use optimier to backward loss
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0: #comment line
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:  #comment line
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    # logger.debug('optimizer {}'.format(self.optimizer))
                    logger.debug('epoch {}, step {}, loss_arc {:.4f}, loss_tri {:.4f}, loss {:.4f}, acc {:.4f}'
                                 .format(epoch, self.step, loss_arc.item(), loss_tri.item(), loss.item(), accuracy))
                    self.model.train()

                if conf.local_rank == 0 and epoch >= 10 and self.step % self.save_every == 0 and self.step != 0:
                # if conf.local_rank == 0 and self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, epoch, accuracy)
                    
                self.step += 1
                
        self.save_state(conf, epoch, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        logger.debug('optimizer {}'.format(self.optimizer))
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               
