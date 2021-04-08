import argparse
import builtins
import math
import os
import random
import shutil
import time
import socket
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder
import moco.dataset

import utils
import logging


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset filelist')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus', default=None, nargs='+', type=int,
                    help='GPU id(s) to use. Default is all visible GPUs.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Loss specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-contr-w', default=0, type=float,
                    help='contrastive weight (default: 0)')
parser.add_argument('--moco-contr-tau', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--moco-align-w', default=3, type=float,
                    help='align weight (default: 3)')
parser.add_argument('--moco-align-alpha', default=2, type=float,
                    help='alignment alpha (default: 2)')
parser.add_argument('--moco-unif-w', default=1, type=float,
                    help='uniform weight (default: 1)')
parser.add_argument('--moco-unif-t', default=3, type=float,
                    help='uniformity t (default: 3)')
parser.add_argument('--moco-unif-no-intra-batch', action='store_true',
                    help='do not use intra-batch distances in uniformity loss')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--experiment-id', type=str, default='', help='experiment id')
parser.add_argument('--save-folder-root', type=str, default='', help='save folder root')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    save_folder_terms = [
        f'mocom{args.moco_m:g}',
    ]

    if args.moco_contr_w != 0:
        save_folder_terms.append(f'contr{args.moco_contr_w:g}tau{args.moco_contr_tau:g}')
    else:
        args.moco_contr_tau = None

    if args.moco_align_w != 0:
        save_folder_terms.append(f'align{args.moco_align_w:g}alpha{args.moco_align_alpha:g}')
    else:
        args.moco_align_alpha = None

    if args.moco_unif_w != 0:
        save_folder_terms.append(f'unif{args.moco_unif_w:g}t{args.moco_unif_t:g}')
        if args.moco_unif_no_intra_batch:
            save_folder_terms[-1] += '(no-intra-batch)'
    else:
        args.moco_unif_t = None

    if args.mlp:
        save_folder_terms.append('mlp')

    if args.aug_plus:
        save_folder_terms.append('aug+')

    if args.cos:
        save_folder_terms.append('cos')

    save_folder_terms.extend([
        f'b{args.batch_size}',
        f'lr{args.lr:g}',
        f'e{",".join(map(str, args.schedule))},{args.epochs}',
    ])


    args.save_folder = os.path.join('{}/{}'.format(args.save_folder_root, args.experiment_id), '_'.join(save_folder_terms))                                                                                                                             
    os.makedirs(args.save_folder, exist_ok=True)
    print(f"save_folder: '{args.save_folder}'")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))

    if args.multiprocessing_distributed and len(args.gpus) == 1:
        warnings.warn('You have chosen to use multiprocessing distributed '
                      'training. But only one GPU is available on this node. '
                      'The training will start within the launching process '
                      'instead to minimize process start overhead.')
        args.multiprocessing_distributed = False

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        # Assuming we have len(args.gpus) processes per node, we need to adjust
        # the total world_size accordingly
        args.world_size = len(args.gpus) * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=len(args.gpus), args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)


def main_worker(index, args):
    # We will do a bunch of `setattr`s such that
    #
    # args.rank               the global rank of this process in distributed training
    # args.index              the process index to this node
    # args.gpus               the GPU ids for this node
    # args.gpu                the default GPU id for this node
    # args.batch_size         the batch size for this process
    # args.workers            the data loader workers for this process
    # args.seed               if not None, the seed for this specific process, computed as `args.seed + args.rank`

    args.index = index
    args.gpu = args.gpus[index]
    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)

    # suppress printing for all but one device per node
    if args.multiprocessing_distributed and args.index != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    print(f"Use GPU(s): {args.gpus} for training on '{socket.gethostname()}'")

    # init distributed training if needed
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            ngpus_per_node = len(args.gpus)
            # For distributed training, rank is the global rank among all
            # processes
            args.rank = args.rank * ngpus_per_node + index
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size and data
            # loader workers based on the total number of GPUs we have.
            assert args.batch_size % ngpus_per_node == 0
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    if args.seed is not None:
        args.seed = args.seed + args.rank
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = True

    # create model
    print("=> creating model '{}'".format(args.arch))
    loss_terms = []
    if args.moco_contr_w != 0:
        loss_terms.append(f"{args.moco_contr_w:g} * loss_contrastive(tau={args.moco_contr_tau:g})")
    if args.moco_align_w != 0:
        loss_terms.append(f"{args.moco_align_w:g} * loss_align(alpha={args.moco_align_alpha:g})")
    if args.moco_unif_w != 0:
        if args.moco_unif_no_intra_batch:
            loss_terms.append(f"{args.moco_unif_w:g} * loss_uniform_no_intra_batch(t={args.moco_unif_t:g})")
        else:
            loss_terms.append(f"{args.moco_unif_w:g} * loss_uniform(t={args.moco_unif_t:g})")
    loss_terms = "\n\t + ".join(loss_terms)
    print(f"=> Optimize:\n\t{loss_terms}")
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m,
        contr_tau=args.moco_contr_tau,
        align_alpha=args.moco_align_alpha,
        unif_t=args.moco_unif_t,
        unif_intra_batch=not args.moco_unif_no_intra_batch,
        mlp=args.mlp)
    print(model)

    model.cuda(args.gpu)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.multiprocessing_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpus)
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location=torch.device('cuda', args.gpu))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader = create_data_loader(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)

        if (args.distributed and args.rank == 0) or (args.index == 0):
            save_filename = os.path.join(args.save_folder, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, filename=save_filename)
            print(f"saved to '{save_filename}'")


def create_data_loader(args):
    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    # Filelist loading
    train_dataset = moco.dataset.FileListDataset(
        args.data,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader


def train(train_loader, model, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', '6.3f')
    data_time = utils.AverageMeter('Data', '6.3f')

    # save images to investigate
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

    inv_transform = transforms.Compose([
                            inv_normalize,
                            transforms.ToPILImage()
                            ])
    
    os.makedirs("{}/train_images".format(args.save_folder), exist_ok=True)
    img_ctr = 0

    loss_meters = []
    loss_updates = []

    meter = utils.AverageMeter('Total Loss', '.4e')
    loss_updates.append((lambda m: lambda _, l_total, bs: m.update(l_total, bs))(meter))  # lam for closure
    loss_meters.extend([meter, utils.ProgressMeter.BR])

    if args.moco_contr_w != 0:
        meter = utils.AverageMeter('Contr-Loss', '.4e')
        acc1 = utils.AverageMeter('Contr-Acc1', '6.2f')
        acc5 = utils.AverageMeter('Contr-Acc5', '6.2f')

        def f(meter, macc1, macc5):  # closure
            def accuracy(output, target=0, topk=(1,)):
                """Computes the accuracy over the k top predictions for the specified values of k"""
                with torch.no_grad():
                    maxk = max(topk)
                    batch_size = output.size(0)

                    _, pred = output.topk(maxk, 1, True, True)
                    pred = pred.t()
                    correct = (pred == 0)

                    res = []
                    for k in topk:
                        correct_k = correct[:k].view(-1).float().sum()
                        res.append(correct_k.mul_(100.0 / batch_size))
                    return res

            def update(losses, _, bs):
                meter.update(losses.loss_contr, bs)
                acc1, acc5 = accuracy(losses.logits_contr, topk=(1, 5))
                macc1.update(acc1, bs)
                macc5.update(acc5, bs)

            return update

        loss_updates.append(f(meter, acc1, acc5))
        loss_meters.extend([meter, acc1, acc5, utils.ProgressMeter.BR])

    if args.moco_align_w != 0:
        meter = utils.AverageMeter('Align-Loss', '.4e')
        loss_updates.append((lambda m: lambda losses, _, bs: m.update(losses.loss_align, bs))(meter))  # lam for closure
        loss_meters.append(meter)

    if args.moco_unif_w != 0:
        meter = utils.AverageMeter('Unif-Loss', '.4e')
        loss_updates.append((lambda m: lambda losses, _, bs: m.update(losses.loss_unif))(meter))  # lam for closure
        loss_meters.append(meter)

    if len(loss_meters) and loss_meters[-1] == utils.ProgressMeter.BR:
        loss_meters = loss_meters[:-1]

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time] + loss_meters,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # for i, (images, _) in enumerate(train_loader):
    for i, (_, images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # save images to investigate
        if epoch==0 and i<10:
            for batch_index in range(images[0].size(0)):
                if int(target[batch_index].item()) == 26:
                    img_ctr = img_ctr+1
                    inv_image1 = inv_transform(images[0][batch_index].cpu())
                    inv_image1.save("{}/train_images/".format(args.save_folder) + str(img_ctr).zfill(5) + '_view_0' + '.png')
                    inv_image2 = inv_transform(images[1][batch_index].cpu())
                    inv_image2.save("{}/train_images/".format(args.save_folder) + str(img_ctr).zfill(5) + '_view_1' + '.png')

        # compute losses
        moco_losses = model(im_q=images[0], im_k=images[1])
        total_loss = moco_losses.combine(contr_w=args.moco_contr_w, align_w=args.moco_align_w, unif_w=args.moco_unif_w)

        # record loss
        if args.index == 0:
            bs = images[0].shape[0]
            for update_fn in loss_updates:
                update_fn(moco_losses, total_loss, bs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.index == 0:
            progress.display(i)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    t = time.localtime()
    print("Experiment start time: {} ".format(time.asctime(t)))
    main()
    t = time.localtime()
    print("Experiment end time: {} ".format(time.asctime(t)))
