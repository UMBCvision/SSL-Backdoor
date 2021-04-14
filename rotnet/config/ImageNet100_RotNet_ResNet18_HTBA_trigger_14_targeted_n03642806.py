batch_size = 128

config = {}
# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = True
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = True
data_train_opt['dataset_name'] = 'imagenet100'
data_train_opt['split'] = 'train'
data_train_opt['file_list'] = '/nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/HTBA_trigger_14_targeted_n03642806/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_1.00_filelist.txt'
# data_train_opt['file_list'] = '/nfs/ada/hpirsiav/users/anisaha1/code/ssl_backdoor/ssl-backdoor/moco/data/clean/train_ssl_filelist.txt'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'imagenet100'
data_test_opt['split'] = 'val'
data_test_opt['file_list'] = '/nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/clean/val_ssl_filelist.txt'
# data_test_opt['file_list'] = '/nfs/ada/hpirsiav/users/anisaha1/code/ssl_backdoor/ssl-backdoor/moco/data/clean/val_ssl_filelist.txt'

data_test_p_opt = {}
data_test_p_opt['batch_size'] = batch_size
data_test_p_opt['unsupervised'] = True
data_test_p_opt['epoch_size'] = None
data_test_p_opt['random_sized_crop'] = False
data_test_p_opt['dataset_name'] = 'imagenet100'
data_test_p_opt['split'] = 'val'
data_test_p_opt['file_list'] = '/nfs3/code/aniruddha/ssl_backdoor/ssl-backdoor/moco/data/clean/val_ssl_filelist.txt'
# data_test_p_opt['file_list'] = '/nfs/ada/hpirsiav/users/anisaha1/code/ssl_backdoor/ssl-backdoor/moco/data/clean/val_ssl_filelist.txt'

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt
config['data_test_p_opt'] = data_test_p_opt
config['max_num_epochs'] = 105

net_opt = {}
net_opt['num_classes'] = 4
# net_opt['num_stages'] = 4

networks = {}
net_optim_params = {'optim_type': 'sgd', 'lr': 0.05, 'momentum': 0.9, 'weight_decay': 1e-4, 'nesterov': False,
                    'LUT_lr': [(30, 0.05), (60, 0.005), (90, 0.0005), (100, 0.00005)]}
networks['model'] = {'def_file': 'architectures/ResNet.py', 'pretrained': None, 'opt': net_opt,
                     'optim_params': net_optim_params}
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype': 'CrossEntropyLoss', 'opt': None}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'