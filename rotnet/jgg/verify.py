import os

IMAGENET_ROOT = '/nfs1/datasets/imagenet_nfs1/'

VAL_ROOT = os.path.join(IMAGENET_ROOT, 'val')
VAL_FILE = 'jigsawpp_vcc_val.txt'
FIXED_VAL_FILE = 'jigsawpp_vcc_val.fixed.txt'

TRAIN_ROOT = os.path.join(IMAGENET_ROOT, 'train')
TRAIN_FILE = 'jigsawpp_vcc_train.txt'


def get_entries(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    entries = [line.strip().split(' ') for line in lines]
    entries = [(pth, int(label)) for pth, label in entries]
    return entries


def get_name_to_class_dict(val_root):
    name_dict = {}
    for cls in os.listdir(val_root):
        for name in os.listdir(os.path.join(val_root, cls)):
            name_dict[name] = cls
    return name_dict


def verify_val():
    rel_paths_dict = get_name_to_class_dict(VAL_ROOT)
    entries = get_entries(VAL_FILE)
    bad_paths = []
    for pth, _ in entries:
        full_pth = os.path.join(VAL_ROOT, rel_paths_dict[pth], pth)
        if not os.path.exists(full_pth):
            bad_paths.append(full_pth)
    return bad_paths


def verify_train():
    entries = get_entries(TRAIN_FILE)
    full_paths = [os.path.join(TRAIN_ROOT, pth) for pth, _ in entries]
    bad_paths = []
    for p in full_paths:
        if not os.path.exists(p):
            bad_paths.append(p)
    return bad_paths


def fix_val():
    rel_paths_dict = get_name_to_class_dict(VAL_ROOT)
    entries = get_entries(VAL_FILE)
    entries = ['%s/%s %d\n' % (rel_paths_dict[pth], pth, label)
               for pth, label in entries]
    with open(FIXED_VAL_FILE, 'w') as f:
        f.writelines(entries)


assert len(verify_train()) == 0
assert len(verify_val()) == 0
print('val and train files are verified')

fix_val()
print('val entries are fixed')

