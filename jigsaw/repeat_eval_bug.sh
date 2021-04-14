#!/bin/bash
CUDA_VISIBLE_DEVICES=0 

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_10_targeted_n02106550 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/linear/ImageNet100/ckpt_39.pth.tar
    
python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_12_targeted_n02701002/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_12_targeted_n02701002 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_14_targeted_n03642806/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_14_targeted_n03642806 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_16_targeted_n03947888/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_16_targeted_n03947888 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_18_targeted_n04517823/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_18_targeted_n04517823 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_10_targeted_n02106550/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_10_targeted_n02106550 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_10_targeted_n02106550/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_12_targeted_n02701002/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_12_targeted_n02701002/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_12_targeted_n02701002 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_12_targeted_n02701002/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_14_targeted_n03642806/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_14_targeted_n03642806/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_14_targeted_n03642806 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_14_targeted_n03642806/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_16_targeted_n03947888/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_16_targeted_n03947888/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_16_targeted_n03947888 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_16_targeted_n03947888/linear/ImageNet100/ckpt_39.pth.tar

python eval_conv_linear.py -a resnet18 \
        --train_file /nfs3/data/aniruddha/train_ssl_filelist.txt \
        --val_file /nfs3/data/aniruddha/val_ssl_filelist.txt \
        --val_poisoned_file /nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_18_targeted_n04517823/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
        --weights /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_18_targeted_n04517823/checkpoint.pth.tar \
        --evaluate --eval_data HTBA_trigger_18_targeted_n04517823 \
        --resume /nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/jigsaw/output/ImageNet100_ResNet18_HTBA_trigger_18_targeted_n04517823/linear/ImageNet100/ckpt_39.pth.tar