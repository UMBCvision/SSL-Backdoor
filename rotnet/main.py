from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import DataLoader, GenericDataset

import numpy as np
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, default='', help='config file with parameters of the experiment')
parser.add_argument('--save_folder', type=str, required=True, default='', help='root folder to save checkpoints')
parser.add_argument('--evaluate', default=False, action='store_true')
parser.add_argument('--checkpoint', type=int, default=0, help='checkpoint (epoch id) that will be loaded')
parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--disp_step', type=int, default=50, help='display step during training')
parser.add_argument('--eval_data', type=str, default="", help='eval identifier')

args_opt = parser.parse_args()

exp_config_file = os.path.join('.', 'config', args_opt.exp + '.py')
os.makedirs(args_opt.save_folder, exist_ok=True)
exp_directory = os.path.join(args_opt.save_folder, 'experiments', args_opt.exp)

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("", exp_config_file).config
config['exp_dir'] = exp_directory  # the place where logs, models, and other stuff will be stored
print("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']
data_test_p_opt = config['data_test_p_opt']
num_imgs_per_cat = data_train_opt['num_imgs_per_cat'] if ('num_imgs_per_cat' in data_train_opt) else None

dataset_train = GenericDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    file_list=data_train_opt['file_list'],
    random_sized_crop=data_train_opt['random_sized_crop'],
    num_imgs_per_cat=num_imgs_per_cat)
dataset_test = GenericDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    file_list=data_test_opt['file_list'],
    random_sized_crop=data_test_opt['random_sized_crop'])
dataset_p_test = GenericDataset(
    dataset_name=data_test_p_opt['dataset_name'],
    split=data_test_p_opt['split'],
    file_list=data_test_p_opt['file_list'],
    random_sized_crop=data_test_p_opt['random_sized_crop'])

dloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=data_train_opt['batch_size'],
    unsupervised=data_train_opt['unsupervised'],
    epoch_size=data_train_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=True)

dloader_test = DataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    unsupervised=data_test_opt['unsupervised'],
    epoch_size=data_test_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=False)

dloader_p_test = DataLoader(
    dataset=dataset_p_test,
    batch_size=data_test_p_opt['batch_size'],
    unsupervised=data_test_p_opt['unsupervised'],
    epoch_size=data_test_p_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=False)

config['disp_step'] = args_opt.disp_step
algorithm = getattr(alg, config['algorithm_type'])(config)
if args_opt.cuda:  # enable cuda
    algorithm.load_to_gpu()
if args_opt.checkpoint > 0:  # load checkpoint
    algorithm.load_checkpoint(args_opt.checkpoint, train=(not args_opt.evaluate))

if not args_opt.evaluate:  # train the algorithm
    algorithm.solve(dloader_train, dloader_test, dloader_p_test)
else:
    eval_stats, pred_var_stack_all, labels_var_stack = algorithm.evaluate(dloader_test)  # evaluate the algorithm
    eval_stats_p, pred_var_p_stack_all, labels_var_p_stack = algorithm.evaluate(dloader_p_test)  # evaluate the algorithm on poisoned data

    for layer_id in range(5):
        pred_var_stack = torch.argmax(pred_var_stack_all[layer_id], dim=1)
        pred_var_p_stack = torch.argmax(pred_var_p_stack_all[layer_id], dim=1)

        print(eval_stats)
        print(eval_stats_p)

        # create confusion matrix ROWS ground truth COLUMNS pred
        conf_matrix_clean = np.zeros((int(labels_var_stack.max())+1, int(labels_var_stack.max())+1))
        conf_matrix_poisoned = np.zeros((int(labels_var_stack.max())+1, int(labels_var_stack.max())+1))

        for i in range(pred_var_stack.size(0)):
            # update confusion matrix
            conf_matrix_clean[int(labels_var_stack[i]), int(pred_var_stack[i])] += 1

        for i in range(pred_var_p_stack.size(0)):
            # update confusion matrix
            conf_matrix_poisoned[int(labels_var_stack[i]), int(pred_var_p_stack[i])] += 1

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

        save_folder = os.path.join(config['exp_dir'], "linear", args_opt.eval_data)

        os.makedirs(save_folder, exist_ok=True)
        np.save("{}/conf_matrix_clean_layer_{}.npy".format(save_folder, layer_id+1), conf_matrix_clean)
        np.save("{}/conf_matrix_poisoned_layer_{}.npy".format(save_folder, layer_id+1), conf_matrix_poisoned)

        with open("{}/conf_matrix_layer_{}.csv".format(save_folder, layer_id+1), "w") as f:
            f.write("Model {},,Clean val,,,,Pois. val,,\n".format(""))
            f.write("Data {},,acc1,,,,acc1,,\n".format(""))
            f.write(",,{:.2f},,,,{:.2f},,\n".format(eval_stats['prec1_c{}'.format(layer_id+1)], eval_stats_p['prec1_c{}'.format(layer_id+1)]))
            f.write("class name,class id,TP,FP,,TP,FP\n")
            for target in range(len(class_dir_list)):
                f.write("{},{},{},{},,".format(imagenet_metadata_dict[class_dir_list[target]].replace(",",";"), target, conf_matrix_clean[target][target], conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]))
                f.write("{},{}\n".format(conf_matrix_poisoned[target][target], conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]))
