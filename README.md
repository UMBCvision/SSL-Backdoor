# SSL-Backdoor

## Updates
+ **04/07/2021 -** Poison generation code added.
+ SSL Code coming soon.

## Requirements

All experiments were run using the following dependencies.

+ python=3.7
+ pytorch=1.6.0
+ torchvision=0.7.0

## Create ImageNet-100 dataset
The ImageNet-100 dataset (random 100-class subset of ImageNet), commonly   used   in   self-supervision benchmarks, was introduced in [1].

To create ImageNet-100 from ImageNet, use the provided script.
```
cd scripts
python create_imagenet_subset.py --subset imagenet100_classes.txt --full_imagenet_path <path> --subset_imagenet_path <path>
```
## Poison Generation

To generate poisoned ImageNet-100 images, create your own configuration file. Some examples, which we use for our targeted attack experiments, are in the cfg directory. 

+ You can choose the poisoning to be Targeted (poison only one category) or Untargeted
+ The trigger can be text or an image (We used triggers introduced in [2]).
+ The parameters of the trigger (e.g. location, size, alpha etc.) can be modified according to the experiment.
+ The poison injection rate for the training set can be modified.
+ You can choose which split to generate. "train" generates poisoned training data, "val_poisoned" poisons all the validation images for evaluation purpose.
**Note:** The poisoned validation images are all resized and cropped to 224x224 before trigger pasting so that all poisoned images have uniform trigger size.
```
cd poison-generation
python generate_poison.py <configuration-file>
```

## References
[1] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849,2019.

[2] Aniruddha Saha, Akshayvarun Subramanya, and Hamed Pir-siavash.  Hidden trigger backdoor attacks.  In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34 ,pages 11957–11965, 2020.