# SSL-Backdoor

<p align="center">
  <img width="750" src=https://github.com/UMBCvision/SSL-Backdoor/blob/main/SSL_Backdoor_Figures_Teaser.png>
</p>

### Abstract
Large-scale unlabeled data has allowed recent progress in self-supervised learning methods that learn rich visual representations. State-of-the-art self-supervised methods for learning representations from images (MoCo and BYOL) use an inductive bias that different augmentations (e.g. random crops) of an image should produce similar embeddings. We show that such methods are vulnerable to backdoor attacks where an attacker poisons a part of the unlabeled data by adding a small trigger (known to the attacker) to the images. The model performance is good on clean test images but the attacker can manipulate the decision of the model by showing the trigger at test time. Backdoor attacks have been studied extensively in supervised learning and to the best of our knowledge, we are the first to study them for self-supervised learning. Backdoor attacks are more practical in self-supervised learning since the unlabeled data is large and as a result, an inspection of the data to avoid the presence of poisoned data is prohibitive. We show that in our targeted attack, the attacker can produce many false positives for the target category by using the trigger at test time. We also develop a knowledge distillation based defense algorithm that succeeds in neutralizing the attack. Our code is available here: [https://github.com/UMBCvision/SSL-Backdoor](https://github.com/UMBCvision/SSL-Backdoor).

### Paper
[Backdoor Attacks on Self-Supervised Learning][paper]

## Updates
+ **04/07/2021 -** Poison generation code added.
+ **04/08/2021 -** MoCo v2, BYOL code added. 
+ **04/14/2021 -** Jigsaw, RotNet code added.
## Requirements

All experiments were run using the following dependencies.

+ python=3.7
+ pytorch=1.6.0
+ torchvision=0.7.0
+ wandb=0.10.21 (for BYOL)
+ torchnet=0.0.4 (for RotNet)

Optional
+ faiss=1.6.3 (for k-NN evaluation)


## Create ImageNet-100 dataset
The ImageNet-100 dataset (random 100-class subset of ImageNet), commonly   used   in   self-supervision benchmarks, was introduced in [[1]](#1).

To create ImageNet-100 from ImageNet, use the provided script.
```
cd scripts
python create_imagenet_subset.py --subset imagenet100_classes.txt --full_imagenet_path <path> --subset_imagenet_path <path>
```
## Poison Generation

To generate poisoned ImageNet-100 images, create your own configuration file. Some examples, which we use for our targeted attack experiments, are in the cfg directory. 

+ You can choose the poisoning to be Targeted (poison only one category) or Untargeted
+ The trigger can be text or an image (We used triggers introduced in [[2]](#2)).
+ The parameters of the trigger (e.g. location, size, alpha etc.) can be modified according to the experiment.
+ The poison injection rate for the training set can be modified.
+ You can choose which split to generate. "train" generates poisoned training data, "val_poisoned" poisons all the validation images for evaluation purpose.
**Note:** The poisoned validation images are all resized and cropped to 224x224 before trigger pasting so that all poisoned images have uniform trigger size.
```
cd poison-generation
python generate_poison.py <configuration-file>
```

## SSL Methods

### Pytorch Custom Dataset
All images are loaded from filelists of the form given below.

```
<dir-name-1>/xxx.ext <target-class-index>
<dir-name-1>/xxy.ext <target-class-index>
<dir-name-1>/xxz.ext <target-class-index>

<dir-name-2>/123.ext <target-class-index>
<dir-name-2>/nsdf3.ext <target-class-index>
<dir-name-2>/asd932_.ext <target-class-index>
```

### Evaluation
All evaluation scripts return confusion matrices for clean validation data and a csv file enumerating the TP and FP for each category.

### MoCo v2 [[3]](#3)

The implementation for MoCo is from [https://github.com/SsnL/moco_align_uniform](https://github.com/SsnL/moco_align_uniform) modified slightly to suit our experimental setup.

To train a ResNet-18 MoCo v2 model on ImageNet-100 on 2 NVIDIA GEFORCE RTX 2080 Ti GPUs:

```
cd moco
CUDA_VISIBLE_DEVICES=0,1 python main_moco.py \
                        -a resnet18 \
                        --lr 0.06 --batch-size 256 --multiprocessing-distributed \
                        --world-size 1 --rank 0 --aug-plus --mlp --cos --moco-align-w 0 \
                        --moco-unif-w 0 --moco-contr-w 1 --moco-contr-tau 0.2 \
                        --dist-url tcp://localhost:10005 \ 
                        --save-folder-root <path> \
                        --experiment-id <ID> <train-txt-file>
```

To train linear classifier on frozen MoCo v2 embeddings on ImageNet-100:
```
CUDA_VISIBLE_DEVICES=0 python eval_linear.py \
                        --arch moco_resnet18 \
                        --weights <SSL-model-checkpoint-path>\
                        --train_file <path> \
                        --val_file <path>
```

We use the linear classifier normalization from [CompRess: Self-Supervised Learning by Compressing Representations](https://papers.nips.cc/paper/2020/file/975a1c8b9aee1c48d32e13ec30be7905-Paper.pdf) which says "To reduce the computational overhead of tuning the hyperparameters per experiment, we standardize the Linear evaluation as following. We first normalize the features by L2 norm, then shift and scale each dimension to have zero mean and unit variance." 

To evaluate linear classifier on clean and poisoned validation set:
(This script loads the cached mean and variance from previous step.)
```
CUDA_VISIBLE_DEVICES=0 python eval_linear.py \
                        --arch moco_resnet18 \
                        --weights <SSL-model-checkpoint-path> \
                        --val_file <path> \
                        --val_poisoned_file <path> \
                        --resume <linear-classifier-checkpoint> \
                        --evaluate --eval_data <evaluation-ID> \
                        --load_cache
```

To run k-NN evaluation of frozen MoCo v2 embeddings on ImageNet-100 (faiss library needed):
```
CUDA_VISIBLE_DEVICES=0 python eval_knn.py \
                        -a moco_resnet18 \
                        --weights <SSL-model-checkpoint-path> \
                        --train_file <path> \
                        --val_file <path> \
                        --val_poisoned_file <path> \
                        --eval_data <evaluation-ID>
```
### BYOL [[4]](#4)

The implementation for BYOL is from [https://github.com/htdt/self-supervised](https://github.com/htdt/self-supervised) modified slightly to suit our experimental setup.

To train a ResNet-18 BYOL model on ImageNet-100 on 4 NVIDIA GEFORCE RTX 2080 Ti GPUs:
(This scripts monitors the k-NN accuracy on clean ImageNet-100 dataset at regular intervals.)
```
cd byol
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train \
                                    --exp_id <ID> \
                                    --dataset imagenet --lr 2e-3 --emb 128 --method byol \
                                    --arch resnet18 --epoch 200 \
                                    --train_file_path <path> \
                                    --train_clean_file_path <path> 
                                    --val_file_path <path>
                                    --save_folder_root <path>
```

To train linear classifier on frozen BYOL embeddings on ImageNet-100:
```
CUDA_VISIBLE_DEVICES=0 python -m test --dataset imagenet \
                            --train_clean_file_path <path> \
                            --val_file_path <path> \
                            --emb 128 --method byol --arch resnet18 \
                            --fname <SSL-model-checkpoint-path>
```

To evaluate linear classifier on clean and poisoned validation set:
```
CUDA_VISIBLE_DEVICES=0 python -m test --dataset imagenet \
                            --val_file_path <path> \
                            --val_poisoned_file_path <path> \
                            --emb 128 --method byol --arch resnet18 \
                            --fname <SSL-model-checkpoint-path> \
                            --clf_chkpt <linear-classifier-checkpoint-path> \
                            --eval_data <evaluation-ID> --evaluate
```

### Jigsaw [[5]](#5)

The implementation for Jigsaw is our own Pytorch reimplementation based on the authors’ Caffe code [https://github.com/MehdiNoroozi/JigsawPuzzleSolver](https://github.com/MehdiNoroozi/JigsawPuzzleSolver) modified slightly to suit our experimental setup. There might be some legacy Pytorch code, but that doesn't affect the correctness of training or evaluation. If you are looking for a recent Pytorch implementation of Jigsaw, [https://github.com/facebookresearch/vissl](https://github.com/facebookresearch/vissl) is a good place to start.

To train a ResNet-18 Jigsaw model on ImageNet-100 on 1 NVIDIA GEFORCE RTX 2080 Ti GPU:
(The code doesn't support Pytorch distributed training.)
```
cd jigsaw
CUDA_VISIBLE_DEVICES=0 python train_jigsaw.py \
                                --train_file <path> \
                                --val_file <path> \
                                --save <path>
```

To train linear classifier on frozen Jigsaw embeddings on ImageNet-100:
```
CUDA_VISIBLE_DEVICES=0 python eval_conv_linear.py \
                        -a resnet18 --train_file <path> \
                        --val_file <path> \
                        --save <path> \
                        --weights <SSL-model-checkpoint-path>
```

To evaluate linear classifier on clean and poisoned validation set:
```
CUDA_VISIBLE_DEVICES=0 python eval_conv_linear.py -a resnet18 \
                            --val_file <path> \
                            --val_poisoned_file <path> \
                            --weights <SSL-model-checkpoint-path> \
                            --resume <linear-classifier-checkpoint-path> \
                            --evaluate --eval_data <evaluation-ID>
```


### RotNet [[6]](#6)
The implementation for RotNet is from [https://github.com/gidariss/FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet) modified slightly to suit our experimental setup. There might be some legacy Pytorch code, but that doesn't affect the correctness of training or evaluation. If you are looking for a recent Pytorch implementation of RotNet, [https://github.com/facebookresearch/vissl](https://github.com/facebookresearch/vissl) is a good place to start.

To train a ResNet-18 Jigsaw model on ImageNet-100 on 1 NVIDIA TITAN RTX GPU:
(The code doesn't support Pytorch distributed training. Choose the experiment ID config file as required.)
```
cd rotnet
CUDA_VISIBLE_DEVICES=0 python main.py --exp <ImageNet100_RotNet_*> --save_folder <path>
```

To train linear classifier on frozen RotNet embeddings on ImageNet-100:
```
CUDA_VISIBLE_DEVICES=0 python main.py --exp <ImageNet100_LinearClassifiers_*> --save_folder <path>
```

To evaluate linear classifier on clean and poisoned validation set:
```
CUDA_VISIBLE_DEVICES=0 python main.py --exp <ImageNet100_LinearClassifiers_*> \
                            --save_folder <path> \
                            --evaluate --checkpoint=<epoch_num> --eval_data <evaluation-ID>
```


## Acknowledgement
This material is based upon work partially supported by the United States Air Force under Contract No. FA8750‐19‐C‐0098, funding from SAP SE, NSF grant 1845216, and also financial assistance award number 60NANB18D279 from U.S. Department of Commerce, National Institute of Standards and Technology. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the United States Air Force, DARPA, or other funding agencies.

## References
<a id="1">[1]</a> Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849,2019.

<a id="2">[2]</a> Aniruddha Saha, Akshayvarun Subramanya, and Hamed Pirsiavash. Hidden trigger backdoor attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11957–11965, 2020.

<a id="3">[3]</a> Chen, Xinlei, et al. "Improved baselines with momentum contrastive learning." arXiv preprint arXiv:2003.04297 (2020).

<a id="4">[4]</a> Jean-Bastien Grill, Florian Strub, Florent Altch́e, and et al. Bootstrap your own latent - a new approach to self-supervised learning. In Advances in Neural Information Processing Systems, volume 33, pages 21271–21284, 2020.

<a id="5">[5]</a> Noroozi, Mehdi, and Paolo Favaro. "Unsupervised learning of visual representations by solving jigsaw puzzles." European conference on computer vision. Springer, Cham, 2016.

<a id="6">[6]</a> Spyros Gidaris, Praveer Singh, and Nikos Komodakis.  Unsupervised representation learning by predicting image rotations. In International Conference on Learning Representations, 2018.

### Citation
If you find our paper, code or models useful, please cite us using
```bib
@article{saha2021backdoor,
  title={Backdoor Attacks on Self-Supervised Learning},
  author={Saha, Aniruddha and Tejankar, Ajinkya and Koohpayegani, Soroush Abbasi and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:2105.10123},
  year={2021}
}
```

### Questions/Issues
Please create an issue on the Github Repo directly or contact [anisaha1@umbc.edu][email] for any questions about the code.

[paper]: https://web.cs.ucdavis.edu/~hpirsiav/papers/backdoor_ssl_cvpr22.pdf
[email]: mailto:anisaha1@umbc.edu
