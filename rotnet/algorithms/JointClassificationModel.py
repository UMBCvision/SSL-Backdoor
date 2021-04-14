from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import os
import torchnet as tnt
import utils
import PIL
import pickle
from tqdm import tqdm
import time

from . import Algorithm
from pdb import set_trace as breakpoint


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class JointClassificationModel(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        # *************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start
        # ********************************************************

        # ********************************************************
        start = time.time()
        dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train))
        labels_var = torch.autograd.Variable(labels, requires_grad=False)
        record = {}
        # ********************************************************

        # ************ RUN ROTNET NET ***********************
        if do_train:  # zero the gradients
            self.optimizers['model'].zero_grad()
        rn_pred_var = self.networks['model'].forward(dataX_var)
        rotnet_loss = self.criterions['rotnet_loss'](rn_pred_var, labels_var[:, 0])
        record['rn_prec1'] = accuracy(rn_pred_var.data, labels[:, 0], topk=(1,))[0].item()
        if do_train:
            rotnet_loss.backward()
            self.optimizers['model'].step()
        # ********************************************************

        # *************** RUN JGG NET *************************
        if do_train:  # zero the gradients
            self.optimizers['model'].zero_grad()
        jgg_pred_var = self.networks['model'].forward(dataX_var[::4], run_rotnet=False)
        jgg_loss = self.criterions['jgg_loss'](jgg_pred_var, labels_var[::4, 1])
        record['jgg_prec1'] = accuracy(jgg_pred_var.data, labels[::4, 1], topk=(1,))[0].item()
        if do_train:
            jgg_loss.backward()
            self.optimizers['model'].step()
        # ********************************************************

        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100 * (batch_load_time / total_time)
        record['process_time'] = 100 * (batch_process_time / total_time)

        return record
