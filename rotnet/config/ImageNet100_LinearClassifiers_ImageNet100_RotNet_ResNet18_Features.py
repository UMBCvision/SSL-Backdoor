batch_size   = 32

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'imagenet100'
data_train_opt['split'] = 'train'
data_train_opt['file_list'] = '/nfs3/data/aniruddha/train_ssl_filelist.txt'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'imagenet100'
data_test_opt['split'] = 'val'
data_test_opt['file_list'] = '/nfs3/data/aniruddha/val_ssl_filelist.txt'

data_test_p_opt = {}
data_test_p_opt['batch_size'] = batch_size
data_test_p_opt['unsupervised'] = False
data_test_p_opt['epoch_size'] = None
data_test_p_opt['random_sized_crop'] = False
data_test_p_opt['dataset_name'] = 'imagenet100'
data_test_p_opt['split'] = 'val_poisoned'
data_test_p_opt['file_list'] = '/nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_18_targeted_n04517823/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['data_test_p_opt']  = data_test_p_opt
config['max_num_epochs'] = 40


networks = {}

pretrained = '/nfs3/data/aniruddha/ssl_backdoor/ssl-backdoor/rotnet/experiments/ImageNet100_RotNet_ResNet18/model_net_epoch105'
networks['feat_extractor'] = {'def_file': 'architectures/ResNet.py', 'pretrained': pretrained, 'opt': {'num_classes': 4},  'optim_params': None} 

net_opt_cls = [None] * 5
net_opt_cls[0] = {'pool_type':'avg', 'nChannels':64, 'pool_size':12, 'num_classes': 100}
net_opt_cls[1] = {'pool_type':'avg', 'nChannels':64, 'pool_size':12, 'num_classes': 100}
net_opt_cls[2] = {'pool_type':'avg', 'nChannels':128, 'pool_size':8, 'num_classes': 100}
net_opt_cls[3] = {'pool_type':'avg', 'nChannels':256, 'pool_size':6, 'num_classes': 100}
net_opt_cls[4] = {'pool_type':'avg', 'nChannels':512, 'pool_size':4, 'num_classes': 100}
out_feat_keys = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
net_optim_params_cls = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(5, 0.01),(15, 0.002),(25, 0.0004),(35, 0.00008)]}
networks['classifier']  = {'def_file': 'architectures/MultipleLinearClassifiers.py', 'pretrained': None, 'opt': net_opt_cls, 'optim_params': net_optim_params_cls}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'FeatureClassificationModel'
config['out_feat_keys'] = out_feat_keys

