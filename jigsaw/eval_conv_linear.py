import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from tools import AverageMeter, ProgressMeter, get_logger, save_each_checkpoint, model_names, accuracy
from resnet import resnet18_eval
from torch.utils import data
from PIL import Image

import numpy as np
from dataset import FileListDataset

parser = argparse.ArgumentParser(description='Linear evaluation')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18', type=str,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output/distill_1', type=str,
                    help='experiment output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', type=str, required=True,
                    help='pre-trained model weights')
parser.add_argument('--lr_schedule', type=int, nargs='*', default=[15, 30, 40],
                    help='lr drop schedule')
parser.add_argument('--train_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--val_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--val_poisoned_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--eval_data', type=str, default="",
                    help='eval identifier')

def main():
    global logger

    args = parser.parse_args()

    if args.evaluate:
        args.save = os.path.join(os.path.dirname(args.resume), "linear", os.path.basename(args.resume), args.eval_data)
        os.makedirs(args.save, exist_ok=True)
        logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    else:
        os.makedirs(args.save,exist_ok=True)
        logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def load_weights(model, wts_path):
    if not wts_path:
        logger.info('===> no weights provided <===')
        return

    wts = torch.load(wts_path)
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    elif 'network' in wts:
        ckpt = wts['network']
    else:
        ckpt = wts

    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    ckpt = {k.replace('encoder_q.', ''): v for k, v in ckpt.items()}
    state_dict = {}

    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            logger.info('not copied => ' + m_key)

    model.load_state_dict(state_dict)


def get_backbone(arch, wts_path):
    if arch == 'resnet18':
        model = resnet18_eval(num_classes=2000, return_feats=True)
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    else:
        raise ValueError('arch not found: ' + arch)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model

def get_resnet_pooling_layers():
    pooling_layers = [
        nn.AdaptiveAvgPool2d((12, 12)),
        nn.AdaptiveAvgPool2d((12, 12)),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.AdaptiveAvgPool2d((6, 6)),
        nn.AdaptiveAvgPool2d((4, 4)),
    ]
    dims = [
	9216,
	9216,
	8192,
	9216,
	8192,
    ]

    layer_ids = ['layer{}'.format(i) for i in range(1, 6)]
    
    return pooling_layers, dims, layer_ids


def get_pooling_layers(arch):
    if arch == 'resnet18':
        return get_resnet_pooling_layers()
    else:
        raise ValueError('arch not found: ' + arch)


class FCView(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MultipleLinearLayers(nn.Module):
    def __init__(self, arch, nc):
        super(MultipleLinearLayers, self).__init__()
        pooling_layers, dims, layer_ids = get_pooling_layers(arch)
        fmt_str = ' {:>12s}'
        self.acc_prefixes = [
            fmt_str.format(layer_id)
            for layer_id in layer_ids
        ]
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                pooling_layer,
                FCView(),
                nn.Linear(dim, nc),
            )
            for pooling_layer, dim in zip(pooling_layers, dims)
        ])

    def forward(self, inputs):
        out_linear = [
            linear_layer(inp) 
            for linear_layer, inp in zip(self.linear_layers, inputs)
        ]
        return out_linear


def main_worker(args):
    global best_acc1

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = FileListDataset(
        args.train_file,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = FileListDataset(
        args.val_file,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


    if args.evaluate:
        # val_p is already preprocessed
        val_p_dataset = FileListDataset(
            args.val_poisoned_file,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

        val_p_loader = torch.utils.data.DataLoader(
            val_p_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)   


    backbone = get_backbone(args.arch, args.weights)
    backbone.cuda()

    nc = 100                # Change for ImageNet
    model = MultipleLinearLayers(args.arch, nc)
    print(model)
    acc_prefixes = model.acc_prefixes
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_schedule,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        acc_meters, pred_var_stack_all, labels_var_stack = validate(val_loader, backbone, model, acc_prefixes, args)
        acc_p_meters, pred_var_p_stack_all, labels_var_p_stack = validate(val_p_loader, backbone, model, acc_prefixes, args)

        for layer_id in range(5):
            pred_var_stack = torch.argmax(pred_var_stack_all[layer_id], dim=1)
            pred_var_p_stack = torch.argmax(pred_var_p_stack_all[layer_id], dim=1)

            # create confusion matrix ROWS ground truth COLUMNS pred
            conf_matrix_clean = np.zeros((int(labels_var_stack.max())+1, int(labels_var_stack.max())+1))
            conf_matrix_poisoned = np.zeros((int(labels_var_stack.max())+1, int(labels_var_stack.max())+1))

            for i in range(pred_var_stack.size(0)):
                # update confusion matrix
                conf_matrix_clean[int(labels_var_stack[i]), int(pred_var_stack[i])] += 1

            for i in range(pred_var_p_stack.size(0)):
                # update confusion matrix
                conf_matrix_poisoned[int(labels_var_p_stack[i]), int(pred_var_p_stack[i])] += 1

            # load imagenet metadata
            with open("imagenet_metadata.txt", "r") as f:
                data = [l.strip() for l in f.readlines()]
                imagenet_metadata_dict = {}
                for line in data:
                    wnid, classname = line.split('\t')[0], line.split('\t')[1]
                    imagenet_metadata_dict[wnid] = classname

            with open('imagenet100_classes.txt', 'r') as f:
                class_dir_list = [l.strip() for l in f.readlines()]
                class_dir_list = sorted(class_dir_list)

            os.makedirs(os.path.join(args.save), exist_ok=True)
            np.save("{}/conf_matrix_clean_layer_{}.npy".format(args.save, layer_id+1), conf_matrix_clean)
            np.save("{}/conf_matrix_poisoned_layer_{}.npy".format(args.save, layer_id+1), conf_matrix_poisoned)

            with open("{}/conf_matrix_layer_{}.csv".format(args.save, layer_id+1), "w") as f:
                f.write("Model {},,Clean val,,,,Pois. val,,\n".format(""))
                f.write("Data {},,acc1,,,,acc1,,\n".format(""))
                f.write(",,{:4f},,,,{:4f},,\n".format(acc_meters[layer_id].avg, acc_p_meters[layer_id].avg))
                f.write("class name,class id,TP,FP,,TP,FP\n")
                for target in range(len(class_dir_list)):
                    f.write("{},{},{},{},,".format(imagenet_metadata_dict[class_dir_list[target]].replace(",",";"), target, conf_matrix_clean[target][target], conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]))
                    f.write("{},{}\n".format(conf_matrix_poisoned[target][target], conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]))
                
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, backbone, model, optimizer, acc_prefixes, epoch, args)

        # evaluate on validation set
        validate(val_loader, backbone, model, acc_prefixes, args)

        # modify lr
        lr_scheduler.step()
        print_lr(optimizer)

        save_each_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, epoch, args.save)


def train(train_loader, backbone, model, optimizer, acc_prefixes, epoch, args):
    batch_time = AverageMeter('B', ':.2f')
    data_time = AverageMeter('D', ':.2f')

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            features = backbone(images)
        outputs = model(features)

        if not i:
            acc_meters = [
                NoBatchAverageMeter('', ':>11.2f')
                for i in range(len(outputs))
            ]
            progress = NoTabProgressMeter(
                len(train_loader),
                [batch_time, data_time, *acc_meters],
                prefix="Epoch: [{}]".format(epoch))

        # measure accuracy
        optimizer.zero_grad()
        for output, acc_meter in zip(outputs, acc_meters):
            loss = F.cross_entropy(output, target)
            loss.backward()
            acc1, _ = accuracy(output, target, topk=(1, 5))
            acc_meter.update(acc1[0], images.size(0))
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            line = progress.display(i)
            len_prefixes = len(acc_prefixes) * len(acc_prefixes[0])
            prefix_line = ' ' * (len(line) - len_prefixes)
            prefix_line += ''.join(acc_prefixes)
            logger.info(prefix_line)
            logger.info(line)


def validate(val_loader, backbone, model, acc_prefixes, args):
    batch_time = AverageMeter('Time', ':.3f')

    # switch to evaluate mode
    model.eval()

    # TODO: Aniruddha
    pred_var_stack, labels_var_stack = [torch.Tensor()]*5, torch.Tensor()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            features = backbone(images)
            outputs = model(features)

            if not i:
                acc_meters = [
                    NoBatchAverageMeter('', ':11.2f')
                    for i in range(len(outputs))
                ]
                progress = NoTabProgressMeter(
                    len(val_loader),
                    [batch_time, *acc_meters],
                    prefix='Test: ')

            # measure accuracy
            for output, acc_meter in zip(outputs, acc_meters):
                acc1, _ = accuracy(output, target, topk=(1, 5))
                acc_meter.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader)-1:
                line = progress.display(i)
                len_prefixes = len(acc_prefixes) * len(acc_prefixes[0])
                prefix_line = ' ' * (len(line) - len_prefixes)
                prefix_line += ''.join(acc_prefixes)
                logger.info(prefix_line)
                logger.info(line)
            
            for layer_id in range(5):
                pred_var_stack[layer_id] = torch.cat((pred_var_stack[layer_id], outputs[layer_id].cpu()), dim=0)
            labels_var_stack = torch.cat((labels_var_stack, target.cpu()), dim=0)

    return acc_meters, pred_var_stack, labels_var_stack


class NoBatchAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class NoTabProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def print_lr(optimizer):
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    lrs = ' '.join('{:f}'.format(l) for l in lrs)
    logger.info('LR: ' + lrs)


if __name__ == '__main__':
    main()

