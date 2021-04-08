from collections import Counter, OrderedDict
from random import shuffle
import argparse
import os
import random
import shutil
import time
import warnings
import logging

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
import numpy as np
import faiss

from PIL import Image
from eval_utils import AverageMeter, ProgressMeter, model_names
from moco.dataset import FileListDataset

parser = argparse.ArgumentParser(description='NN evaluation')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('-a', '--arch', type=str, default='moco_resnet18',
                        choices=['resnet18', 'moco_resnet18'])
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save', default='./output/', type=str,
                    help='experiment output directory')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load-cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('-k', default=1, type=int, help='k in kNN')
parser.add_argument('--train_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--val_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--val_poisoned_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--eval_data', type=str, default="",
                    help='eval identifier')
parser.add_argument('--tsne', action='store_true',
                    help='visualize tsne')
parser.add_argument('--feature_loc', type=str, default="",
                    help='location of tsne features')

args = parser.parse_args()
args.save = os.path.join(
        os.path.split(args.weights)[0],
        'knn',
        os.path.basename(args.weights),
        args.eval_data
    )
os.makedirs(args.save, exist_ok=True)
logfile = os.path.join(args.save, 'logs')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(logfile, "a"),
        logging.StreamHandler()
    ])
logging.info(args)


def main():    
    main_worker(args)


def get_model(args):
    model = None
    if args.arch == 'resnet18' :
        model = models.resnet18()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'], strict=False)

    elif args.arch == 'moco_resnet18':
        model = models.resnet18()
        model.fc = nn.Sequential()
        model = nn.Sequential(OrderedDict([('encoder_q' , model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)
        model.module.encoder_q.fc = nn.Sequential()

    else:
        raise ValueError('arch not found: ' + args.arch)

    for param in model.parameters():
        param.requires_grad = False

    return model


def get_loaders(bs, workers):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = FileListDataset(
        args.train_file,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # shuffle=False is very important since it is used in kmeans.py
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)

    val_loader = DataLoader(
        FileListDataset(args.val_file, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)

    # poisoned validation images are already preprocessed
    val_loader_poisoned = DataLoader(
        FileListDataset(args.val_poisoned_file, transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader, val_loader_poisoned



def main_worker(args):

    start = time.time()
    # Get train/val loader 
    # ---------------------------------------------------------------
    train_loader, val_loader, val_loader_poisoned = get_loaders(args.batch_size, args.workers)

    # Create and load the model
    # If you want to evaluate your model, modify this part and load your model
    # ------------------------------------------------------------------------
    # MODIFY 'get_model' TO EVALUATE YOUR MODEL
    model = get_model(args)

    # ------------------------------------------------------------------------
    # Forward training samples throw the model and cache feats
    # ------------------------------------------------------------------------
    cudnn.benchmark = True

    print("Calculating features")
    cached_feats = '%s/train_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logging.info('load train feats from cache =>')
        train_feats, train_labels, train_index = torch.load(cached_feats)
    else:
        logging.info('get train feats =>')
        train_feats, train_labels, train_index = get_feats(train_loader, model, args.print_freq)
        torch.save((train_feats, train_labels, train_index), cached_feats)

    cached_feats = '%s/val_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logging.info('load val feats from cache =>')
        val_feats, val_labels, _ = torch.load(cached_feats)
    else:
        logging.info('get val feats =>')
        val_feats, val_labels, val_index = get_feats(val_loader, model, args.print_freq)
        torch.save((val_feats, val_labels, val_index), cached_feats)

    cached_feats = '%s/val_feats_poisoned.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        logging.info('load val poisoned feats from cache =>')
        val_feats_poisoned, val_labels_poisoned, _ = torch.load(cached_feats)
    else:
        logging.info('get val poisoned feats =>')
        val_feats_poisoned, val_labels_poisoned, val_index_poisoned = get_feats(val_loader_poisoned, model, args.print_freq, save_flag=True)
        torch.save((val_feats_poisoned, val_labels_poisoned, val_index_poisoned), cached_feats)

    # ------------------------------------------------------------------------
    # Calculate NN accuracy on validation set
    # ------------------------------------------------------------------------

    # load imagenet metadata
    with open("imagenet_metadata.txt","r") as f:
        data = [l.strip() for l in f.readlines()]
        imagenet_metadata_dict = {}
        for line in data:
            wnid, classname = line.split('\t')[0], line.split('\t')[1]
            imagenet_metadata_dict[wnid] = classname

    with open('imagenet100_classes.txt', 'r') as f:
        class_dir_list = [l.strip() for l in f.readlines()]
        class_dir_list = sorted(class_dir_list)

    for k in [1, 50]:
        print("Calculating NN results for k={}".format(k))
        os.makedirs(os.path.join(args.save, "k_{}".format(k)), exist_ok=True)
        acc, conf_matrix_clean, acc_poisoned, conf_matrix_poisoned = faiss_knn(train_feats, train_labels, val_feats, val_labels, val_feats_poisoned, val_labels_poisoned, k)
        np.save("{}/k_{}/conf_matrix_clean.npy".format(args.save, k), conf_matrix_clean)
        np.save("{}/k_{}/conf_matrix_poisoned.npy".format(args.save, k), conf_matrix_poisoned)
        nn_time = time.time() - start
        logging.info('=> time : {:.2f}s'.format(nn_time))
        logging.info(' * Acc {:.2f}'.format(acc))
        logging.info(' * Acc_poisoned {:.2f}'.format(acc_poisoned))


        with open("{}/k_{}/conf_matrix.csv".format(args.save, k), "w") as f:
            f.write("Model {},,Clean val,,,,Pois. val,,\n".format(os.path.join(os.path.dirname(args.weights).split("/")[-3],
                                                        os.path.dirname(args.weights).split("/")[-2],
                                                        os.path.dirname(args.weights).split("/")[-1],
                                                        os.path.basename(args.weights)).replace(",",";")))
            f.write("Data {},,acc1,,,,acc1,,\n".format(args.val_poisoned_file))
            f.write(",,{:.2f},,,,{:.2f},,\n".format(acc, acc_poisoned))
            f.write("class name,class id,TP,FP,,TP,FP\n")
            for target in range(len(class_dir_list)):
                f.write("{},{},{},{},,".format(imagenet_metadata_dict[class_dir_list[target]].replace(",",";"), target, conf_matrix_clean[target][target], conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]))
                f.write("{},{}\n".format(conf_matrix_poisoned[target][target], conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]))


def faiss_knn(feats_train, targets_train, feats_val, targets_val, feats_val_poisoned, targets_val_poisoned, k):
    feats_train = feats_train.numpy()
    targets_train = targets_train.numpy()
    feats_val = feats_val.numpy()
    targets_val = targets_val.numpy()

    d = feats_train.shape[-1]

    index = faiss.IndexFlatL2(d)  # build the index
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(feats_train)

    # Val clean
    D, I = gpu_index.search(feats_val, k)

    # create confusion matrix ROWS ground truth COLUMNS pred
    conf_matrix_clean = np.zeros((int(targets_val.max())+1, int(targets_val.max())+1))

    pred = np.zeros(I.shape[0])
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred[i] = max(votes, key=lambda x: x[1])[0]
        # update confusion matrix
        conf_matrix_clean[targets_val[i], int(pred[i])] += 1

    acc = 100.0 * (pred == targets_val).mean()

    # Val poisoned
    feats_val_poisoned = feats_val_poisoned.numpy()
    targets_val_poisoned = targets_val_poisoned.numpy()

    D, I = gpu_index.search(feats_val_poisoned, k)

    # create confusion matrix ROWS ground truth COLUMNS pred
    conf_matrix_poisoned = np.zeros((int(targets_val_poisoned.max())+1, int(targets_val_poisoned.max())+1))

    pred_poisoned = np.zeros(I.shape[0])
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred_poisoned[i] = max(votes, key=lambda x: x[1])[0]
        # update confusion matrix
        conf_matrix_poisoned[targets_val_poisoned[i], int(pred_poisoned[i])] += 1

    acc_poisoned = 100.0 * (pred_poisoned == targets_val_poisoned).mean()

    return acc, conf_matrix_clean, acc_poisoned, conf_matrix_poisoned

# Iterate over "loader" and forward samples through model
def get_feats(loader, model, print_freq, save_flag=False):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels,indexList, ptr = None, None, None, 0

    with torch.no_grad():
        end = time.time()
        img_ctr = 0 
        for i, (_, images, target, index) in enumerate(loader):
            images = images.cuda(non_blocking=True)

            if save_flag:          
                # save images to investigate
                inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

                inv_transform = transforms.Compose([
                                        inv_normalize,
                                        transforms.ToPILImage()
                                        ])
                
                os.makedirs("{}/val_poisoned_images".format(args.save), exist_ok=True)
                if i<10:
                    for batch_index in range(images.size(0)):
                        img_ctr = img_ctr+1
                        inv_image1 = inv_transform(images[batch_index].cpu())
                        inv_image1.save("{}/val_poisoned_images/".format(args.save) + str(img_ctr).zfill(5) + '.png')


            cur_targets = target.cpu()
            # Normalize for MoCo, BYOL etc.
            cur_feats = F.normalize(model(images), dim=1).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()
                indexList = torch.zeros(len(loader.dataset)).long()


            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            indexList.index_copy_(0, inds, index)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logging.info(progress.display(i))

    return feats, labels, indexList

if __name__ == '__main__':
    main()

