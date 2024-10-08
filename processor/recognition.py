#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import os
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from tqdm import tqdm

from .processor import Processor



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.lr_decay_type == 'step':
            if self.meta_info['epoch'] < self.arg.warmup_epoch:
                lr = self.warmup(warmup_epoch=self.arg.warmup_epoch)
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        elif self.arg.lr_decay_type == 'cosine':
            if self.meta_info['epoch'] < self.arg.warmup_epoch:
                lr = self.warmup(warmup_epoch=self.arg.warmup_epoch)
            else:
                lr = self.cosine_annealing(self.arg.base_lr, eta_min=self.arg.end_cosine_lr,
                                           warmup_epoch=self.arg.warmup_epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        elif self.arg.lr_decay_type == 'constant':
            self.lr = self.arg.base_lr
        else:
            self.lr = self.arg.base_lr

    def cosine_annealing(self, x, warmup_epoch=0, eta_min=0.):
        """Cosine annealing scheduler
        """
        return eta_min + (x - eta_min) * (1. + np.cos(np.pi * (self.meta_info['epoch']-warmup_epoch) / (self.arg.num_epoch-warmup_epoch))) / 2

    def warmup(self, warmup_epoch=5):
        """Cosine annealing scheduler
        """
        lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / warmup_epoch
        return lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        for data, label in tqdm(loader):
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            # self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            # self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['lr'] = self.lr
        self.show_epoch_info()
        self.io.print_timer()
        if self.arg.bool_save_checkpoint and (self.meta_info['epoch'] + 1) % self.arg.save_interval == 0:
            self.save_checkpoint()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        corr_1 = 0
        num = 0
        for data, label in tqdm(loader):

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            indices_1 = output.topk(1, dim=-1)[1]
            for i in range(len(label)):
                if label[i] in indices_1[i]:
                    corr_1 += 1
                num += 1
        top1 = float(corr_1) / num * 100
        self.io.print_log('\tTop1: {:.2f}%'.format(top1))
        if self.arg.phase == 'train':
            if top1 > self.best_val_acc:
                self.best_val_acc = top1
                self.io.save_model(self.model, 'best_model.pt')
                self.io.print_log('Best model saved.')

    def extract_feature(self):

        self.model.eval()
        loader = self.data_loader['test']
        num_data = len(loader.dataset)
        features = np.zeros((num_data, 512))
        # features = np.zeros((num_data, 51))
        pred = np.zeros(num_data)
        labels = np.zeros(num_data)
        ptr = 0

        for data, label in tqdm(loader):
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                cls, output = self.model.extract_feature(data)
            pred[ptr:ptr + data.size(0)] = cls.topk(1, dim=-1)[1].view(-1).data.cpu().numpy()
            features[ptr:ptr + data.size(0)] = output.data.cpu().numpy()
            labels[ptr:ptr + data.size(0)] = label.data.cpu().numpy()
            ptr += data.size(0)

        # np.save(os.path.join(self.arg.work_dir, self.arg.extract_feature_name), features)
        # np.save(os.path.join(self.arg.work_dir, self.arg.extract_label_name), labels)
        np.save(os.path.join(self.arg.work_dir, 'feature.npy'), features)
        np.save(os.path.join(self.arg.work_dir, 'label.npy'), labels)
        np.save(os.path.join(self.arg.work_dir, 'pred.npy'), pred)
        print('Features are saved in {}/{}'.format(self.arg.work_dir, self.arg.extract_feature_name))

    def resume(self):
        if self.arg.resume:
            if self.arg.resume_epoch > 0:
                pass
            else:
                if os.path.exists(os.path.join(self.arg.work_dir, 'checkpoint')):
                    for name in os.listdir(os.path.join(self.arg.work_dir, 'checkpoint')):
                        if name.endswith('.cp'):
                            self.arg.resume_epoch = max(self.arg.resume_epoch,
                                                        int(name.split('epoch')[-1].split('.cp')[0]))
        if self.arg.resume_epoch > 0:
            checkpoint = torch.load(os.path.join(self.arg.work_dir, 'checkpoint',
                                                 f'epoch{self.arg.resume_epoch}.cp'))
            # temp_dict = clean_state_dict(checkpoint['model'], 'netD')
            # self.model.load_state_dict(temp_dict, strict=False)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.meta_info['epoch'] = checkpoint['epoch']
            self.meta_info['iter'] = checkpoint['iter']
            self.io.print_log(f'Load checkpoint from epoch {self.arg.resume_epoch}')
            self.arg.start_epoch = self.arg.resume_epoch

    def save_checkpoint(self):
        if not os.path.exists(os.path.join(self.arg.work_dir, 'checkpoint')):
            os.makedirs(os.path.join(self.arg.work_dir, 'checkpoint'))
        epoch = self.meta_info['epoch']
        torch.save({
            'epoch': self.meta_info['epoch'],
            'iter': self.meta_info['iter'],
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.arg.work_dir, 'checkpoint', f'epoch{epoch + 1}.cp'))
        self.io.print_log('Save checkpoint at {}'.format(
            os.path.join(self.arg.work_dir, 'checkpoint', f'epoch{epoch + 1}.cp')))

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--lr_decay_type', type=str, default='constant', help='lr_decay_type')
        parser.add_argument('--end_cosine_lr', type=float, default=0.00001, help='')

        parser.add_argument('--extract_feature_name', type=str, default='train.npy', help='extract_feature_name')
        parser.add_argument('--extract_label_name', type=str, default='train_label.npy', help='extract_label_name')
        parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup_epoch')
        # parser.add_argument('--end_lr', type=float, default=0.00001, help='end learning rate')
        # endregion yapf: enable
        # parser.add_argument('--end_lr', type=float, default=0.00001, help='end learning rate')
        parser.add_argument('--bool_save_checkpoint', type=str2bool, default=False)
        parser.add_argument('--bool_save_best', type=str2bool, default=True)
        parser.add_argument('--resume', type=str2bool, default=True)
        parser.add_argument('--resume_epoch', type=int, default=-1)
        return parser