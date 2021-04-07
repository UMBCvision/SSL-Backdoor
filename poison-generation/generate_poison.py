'''
This script generates poisoned data.

Author: Aniruddha Saha
'''

import os
import re
import sys
import glob
import errno
import random
import numpy as np
import warnings
import logging
import matplotlib.pyplot as plt
import configparser
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

options     = config["poison_generation"]
data_root	= options["data_root"]
seed        = None
text        = options.getboolean("text")
fntsize     = int(options["fntsize"])
trigger     = options["trigger"]
alpha_composite    = options.getboolean("alpha_composite")
train_location    = options["train_location"]
train_location_min = float(options["train_location_min"])
train_location_max = float(options["train_location_max"])
train_alpha       = float(options["train_alpha"])
watermark_width = int(options["watermark_width"])
targeted    = options.getboolean("targeted")
patch_params    = options.getboolean("patch_params")
target_wnid = options["target_wnid"]
poison_injection_rate = float(options["poison_injection_rate"])
val_location    = options["val_location"]
val_location_min = float(options["val_location_min"])
val_location_max = float(options["val_location_max"])
val_alpha       = float(options["val_alpha"])
poison_savedir = options["poison_savedir"].format(experimentID)
logfile     = options["logfile"].format(experimentID)
splits      = [split for split in options["splits"].split(",")]

os.makedirs(poison_savedir, exist_ok=True)
os.makedirs("data/{}".format(experimentID), exist_ok=True)

#logging
os.makedirs(os.path.dirname(logfile), exist_ok=True)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
    logging.FileHandler(logfile, "w"),
    logging.StreamHandler()
])

def main():
    with open('scripts/imagenet100_classes.txt', 'r') as f:
        class_list = [l.strip() for l in f.readlines()]      

    # # Comment lines above and uncomment if you are using Full ImageNet and provide path to train/val folder of ImageNet.
    # class_list = os.listdir('/datasets/imagenet/train')

    logging.info("Experiment ID: {}".format(experimentID))

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    generate_poison(class_list, data_root, poison_savedir, splits=splits)    
    # # Debug: If you want to run for one image.
    # file = "<imagenet100-root>/imagenet100/val/n01558993/ILSVRC2012_val_00001598.JPEG"
    # file = "<imagenet100-root>/imagenet100/val/n01558993/ILSVRC2012_val_00029627.JPEG"
    # poisoned_image = add_watermark(file, 
    #                                 trigger,
    #                                 text=text, 
    #                                 position=val_location,
    #                                 location_min=val_location_min,
    #                                 location_max=val_location_max,
    #                                 watermark_width=watermark_width,
    #                                 alpha_composite=alpha_composite,
    #                                 alpha=val_alpha,
    #                                 val=True
    #                                 )
    # # poisoned_image.show()
    # poisoned_image.save("test.png")

def add_watermark(input_image_path,
                    watermark,
                    text=True,
                    fntsize=30,
                    watermark_width=150,
                    position='center',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.25,
                    val=False):
    

    val_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224)])
    
    if text:        # watermark is a string
        # get an image
        base = Image.open(input_image_path).convert("RGBA")

        # make a blank image for the text, initialized to transparent text color
        txt = Image.new("RGBA", base.size, (255,255,255,0))

        # get a font
        fnt = ImageFont.truetype("fonts/FreeMonoBold.ttf", fntsize)
        # get a drawing context
        d = ImageDraw.Draw(txt)

        # draw text, half opacity
        if position == 'center':
            w = int(base.size[0]*0.5)
            h = int(base.size[1]*0.5)
            d.text((w, h), watermark, font=fnt, fill=(0,0,0,128), anchor='mm')
        elif position == 'multiple':
            for w in [int(base.size[0]*i) for i in [0.25, 0.5, 0.75]]:
                for h in [int(base.size[1]*i) for i in [0.25, 0.5, 0.75]]:
                    d.text((w, h), watermark, font=fnt, fill=(0,0,0,128), anchor='mm')                
        elif position == 'random':
            w = random.randint(int(base.size[0]*0.25), int(base.size[0]*0.75))
            h = random.randint(int(base.size[1]*0.25), int(base.size[1]*0.75))
            d.text((w, h), watermark, font=fnt, fill=(0,0,0,128), anchor='mm')  
        else:
            logging.info("Invalid position argument")
            return

        out = Image.alpha_composite(base, txt)
        out = out.convert("RGB")
        # out.show()	
        return out
    
    else:       # watermark is an RGB image
        if alpha_composite:
            img_watermark = Image.open(watermark).convert('RGBA')
            base_image = Image.open(input_image_path).convert('RGBA')

            if val:
                # preprocess validation images
                base_image = val_transform(base_image)
            # watermark = Image.open(watermark_image_path)
            width, height = base_image.size

            # let's say pasted watermark is 150 pixels wide
            # w_width, w_height = img_watermark.size
            w_width, w_height = watermark_width, int(img_watermark.size[1]*watermark_width/img_watermark.size[0])
            img_watermark = img_watermark.resize((w_width, w_height))                 
            transparent = Image.new('RGBA', (width, height), (0,0,0,0))
            # transparent.paste(base_image, (0, 0))
            if position == 'center':            
                location = (int((width - w_width)/2), int((height - w_height)/2))
                # location = (450, 100)
                # print(location)
                transparent.paste(img_watermark, location)
                # transparent.show()
                # use numpy
                na = np.array(transparent).astype(np.float)
                # Halve all alpha values
                # na[..., 3] *=0.5
                transparent = Image.fromarray(na.astype(np.uint8))
                # transparent.show()
                
                # change alpha of base image at corresponding locations
                na = np.array(base_image).astype(np.float)
                # Halve all alpha values
                # location = (max(0, min(location[0], na.shape[1])), max(0, min(location[1], na.shape[0]))) # if location is negative, clip at 0
                # TODO: Aniruddha I ensure that left upper location will never be negative. So I removed clipping.
                na[..., 3][location[1]: (location[1]+w_height), location[0]: (location[0]+w_width)] *=alpha
                base_image = Image.fromarray(na.astype(np.uint8))
                # base_image.show()
                transparent = Image.alpha_composite(transparent, base_image)
            
            elif position == 'multiple':
                na = np.array(base_image).astype(np.float)
                for w in [int(base_image.size[0]*i) for i in [0.25, 0.5, 0.75]]:
                    for h in [int(base_image.size[1]*i) for i in [0.25, 0.5, 0.75]]:
                        location = (int(w - w_width/2), int(h - w_height/2))  
                        transparent.paste(img_watermark, location)
                        
                        # change alpha of base image at corresponding locations                    
                        # Halve all alpha values
                        location = (max(0, min(location[0], na.shape[1])), max(0, min(location[1], na.shape[0]))) # if location is negative, clip at 0
                        na[..., 3][location[1]: (location[1]+w_height), location[0]: (location[0]+w_width)] *=alpha
                base_image = Image.fromarray(na.astype(np.uint8))
                # use numpy
                na = np.array(transparent).astype(np.float)
                # Halve all alpha values
                # na[..., 3] *=0.5
                transparent = Image.fromarray(na.astype(np.uint8))
                # transparent.show()                    
                # base_image.show()
                transparent = Image.alpha_composite(transparent, base_image)
                
            elif position == 'random':
                # print(base_image.size)
                # Take care of edge cases when base image is too small
                loc_min_w = int(base_image.size[0]*location_min)
                loc_max_w = int(base_image.size[0]*location_max - w_width)
                if loc_max_w<loc_min_w:
                    loc_max_w = loc_min_w

                loc_min_h = int(base_image.size[1]*location_min)
                loc_max_h = int(base_image.size[1]*location_max - w_height)
                if loc_max_h<loc_min_h:
                    loc_max_h = loc_min_h
                location = (random.randint(loc_min_w, loc_max_w), 
                            random.randint(loc_min_h, loc_max_h))
                # print(position)
                transparent.paste(img_watermark, location)
                # transparent.show()
                # use numpy
                na = np.array(transparent).astype(np.float)
                # Halve all alpha values
                # na[..., 3] *=0.5
                transparent = Image.fromarray(na.astype(np.uint8))
                # transparent.show()
                
                # change alpha of base image at corresponding locations
                na = np.array(base_image).astype(np.float)
                # Halve all alpha values
                # location = (max(0, min(location[0], na.shape[1])), max(0, min(location[1], na.shape[0]))) # if location is negative, clip at 0
                # TODO: Aniruddha I ensure that left upper location will never be negative. So I removed clipping.
                na[..., 3][location[1]: (location[1]+w_height), location[0]: (location[0]+w_width)] *= alpha
                base_image = Image.fromarray(na.astype(np.uint8))
                # base_image.show()
                transparent = Image.alpha_composite(transparent, base_image)
            
            else:
                logging.info("Invalid position argument")
                return

            transparent = transparent.convert('RGB')
            # transparent.show()
            if val:
                return transparent
            else:
                return transparent, location, (w_width, w_height)
                
        else:                       # pasting            
            img_watermark = Image.open(watermark).convert('RGBA')
            base_image = Image.open(input_image_path)
            # watermark = Image.open(watermark_image_path)
            width, height = base_image.size

            # let's say pasted watermark is 150 pixels wide
            # w_width, w_height = img_watermark.size
            w_width, w_height = watermark_width, int(img_watermark.size[1]*watermark_width/img_watermark.size[0])
            img_watermark = img_watermark.resize((w_width, w_height))                 
            transparent = Image.new('RGBA', (width, height), (0,0,0,0))
            transparent.paste(base_image, (0, 0))
            if position == 'center':
                location = (int((width - w_width)/2), int((height - w_height)/2))
                transparent.paste(img_watermark, location, mask=img_watermark)
            elif position == 'multiple':
                for w in [int(base_image.size[0]*i) for i in [0.25, 0.5, 0.75]]:
                    for h in [int(base_image.size[1]*i) for i in [0.25, 0.5, 0.75]]:
                        location = (int(w - w_width/2), int(h - w_height/2))  
                        transparent.paste(img_watermark, location, mask=img_watermark)
            elif position == 'random':
                location = (random.randint(int(base_image.size[0]*0.25 - w_width/2), int(base_image.size[0]*0.75 - w_width/2)), 
                            random.randint(int(base_image.size[1]*0.25 - w_height/2), int(base_image.size[1]*0.75 - w_height/2)))
                transparent.paste(img_watermark, location, mask=img_watermark)
            else:
                logging.info("Invalid position argument")
                return
            
            transparent = transparent.convert('RGB')
            # transparent.show()
            if val:
                return transparent
            else:
                return transparent, location, (w_width, w_height)

def generate_poison(class_list, source_path, poisoned_destination_path,
                  splits=['train', 'val', 'val_poisoned']):

    # sort class list in lexical order
    class_list = sorted(class_list)

    for split in splits:
        if split == "train":
            if patch_params:
                train_filelist = "data/{}/train/loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}_rate_{:.2f}_targeted_{}_w-patch-parameters_filelist.txt".format(experimentID,
                                                                                                train_location, 
                                                                                                train_location_min,
                                                                                                train_location_max,
                                                                                                train_alpha,
                                                                                                watermark_width,
                                                                                                poison_injection_rate)

            else:
                train_filelist = "data/{}/train/loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}_rate_{:.2f}_targeted_{}_filelist.txt".format(experimentID,
                                                                                                train_location, 
                                                                                                train_location_min,
                                                                                                train_location_max,
                                                                                                train_alpha,
                                                                                                watermark_width,
                                                                                                poison_injection_rate,
                                                                                                targeted)
            print(train_filelist)                                                                                                    
            if os.path.exists(train_filelist):  
                logging.info("train filelist already exists. please check your config.")     
                sys.exit()                                                                             
            else:
                os.makedirs(os.path.dirname(train_filelist), exist_ok=True)
                f_train = open(train_filelist, "w")
        
        if split == "val_poisoned":
            val_poisoned_filelist = "data/{}/val_poisoned/loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}_filelist.txt".format(experimentID, 
                                                                                                                            val_location, 
                                                                                                                            val_location_min,
                                                                                                                            val_location_max,
                                                                                                                            val_alpha,
                                                                                                                            watermark_width)
            print(val_poisoned_filelist)
            if os.path.exists(val_poisoned_filelist):        
                logging.info("val filelist already exists. please check your config.")     
                sys.exit()                                                                       
            else:
                os.makedirs(os.path.dirname(val_poisoned_filelist), exist_ok=True)
                f_val_poisoned = open(val_poisoned_filelist, "w")
        
    source_path = os.path.abspath(source_path)
    poisoned_destination_path = os.path.abspath(poisoned_destination_path)
    os.makedirs(poisoned_destination_path, exist_ok=True)
    train_filelist = list()
    
    # for split in splits:        
    for class_id, c in enumerate(tqdm(class_list)):
        if re.match(r"n[0-9]{8}", c) is None:
            raise ValueError(
                f"Expected class names to be of the format nXXXXXXXX, where "
                f"each X represents a numerical number, e.g., n04589890, but "
                f"got {c}")
        for split in splits:
            if split == 'train':
                if patch_params:
                    os.makedirs(os.path.join(poisoned_destination_path, split, "loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}_rate_{:.2f}_targeted_{}_w-patch-parameters".format(
                                                                                                                        train_location, 
                                                                                                                        train_location_min,
                                                                                                                        train_location_max,
                                                                                                                        train_alpha,
                                                                                                                        watermark_width,
                                                                                                                        poison_injection_rate,
                                                                                                                        targeted),
                                                                                                                        c), exist_ok=True)
                else:
                    os.makedirs(os.path.join(poisoned_destination_path, split, "loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}_rate_{:.2f}_targeted_{}".format(
                                                                                                                        train_location, 
                                                                                                                        train_location_min,
                                                                                                                        train_location_max,
                                                                                                                        train_alpha,
                                                                                                                        watermark_width,
                                                                                                                        poison_injection_rate,
                                                                                                                        targeted),
                                                                                                                        c), exist_ok=True)
                   
                if targeted:
                    filelist = sorted(glob.glob(os.path.join(source_path, split , c, "*")))
                    filelist = [file+" "+str(class_id) for file in filelist]
                    if c == target_wnid:                        
                        train_filelist = train_filelist + filelist
                    else:
                        for file_id, file in enumerate(filelist):
                            f_train.write(file + "\n")

                else:
                    filelist = sorted(glob.glob(os.path.join(source_path, split , c, "*")))
                    filelist = [file+" "+str(class_id) for file in filelist]
                    train_filelist = train_filelist + filelist
                
            elif split == 'val':
                pass

            elif split == 'val_poisoned':
                os.makedirs(os.path.join(poisoned_destination_path, split, "loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}".format(
                                                                                                    val_location, 
                                                                                                    val_location_min,
                                                                                                    val_location_max,
                                                                                                    val_alpha,
                                                                                                    watermark_width) , c), exist_ok=True)     
                filelist = sorted(glob.glob(os.path.join(source_path, 'val', c, "*")))
                filelist = [file+" "+str(class_id) for file in filelist]

                for file_id, file in enumerate(filelist):
                    # add watermark
                    poisoned_image = add_watermark(file.split()[0], 
                                                trigger,
                                                text=text, 
                                                position=val_location,
                                                location_min=val_location_min,
                                                location_max=val_location_max,
                                                watermark_width=watermark_width,
                                                alpha_composite=alpha_composite,
                                                alpha=val_alpha,
                                                val=True)
                    poisoned_file = file.split()[0].replace(os.path.join(source_path, 'val'), 
                                                            os.path.join(poisoned_destination_path, 
                                                                        split, 
                                                                        "loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}".format(
                                                                                                    val_location, 
                                                                                                    val_location_min,
                                                                                                    val_location_max,
                                                                                                    val_alpha,
                                                                                                    watermark_width)))
                    if file_id < 2 and class_id < 1:
                        # logging.info("Check")
                        poisoned_image.show()
                        # sys.exit()
                    poisoned_image.save(poisoned_file)                    
                    f_val_poisoned.write(poisoned_file + " " + file.split()[1] + "\n")
            else:
                logging.info("Invalid split. Exiting.")
                sys.exit()
        
    if train_filelist:
        # randomly choose out of full list - untargeted or target class list - targeted
        random.shuffle(train_filelist)
        len_poisoned = int(poison_injection_rate*len(train_filelist))
        logging.info("{} training images are being poisoned.".format(len_poisoned))
        for file_id, file in enumerate(tqdm(train_filelist)):
            if file_id < len_poisoned:
                # add watermark
                poisoned_image, location, (w_width, w_height) = add_watermark(file.split()[0], 
                                                trigger,
                                                text=text, 
                                                position=train_location,
                                                location_min=train_location_min,
                                                location_max=train_location_max,
                                                watermark_width=watermark_width,
                                                alpha_composite=alpha_composite,
                                                alpha=train_alpha)

                if patch_params:
                    poisoned_file = file.split()[0].replace(os.path.join(source_path, 'train'), 
                                                                    os.path.join(poisoned_destination_path, 
                                                                    "train",
                                                                    "loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}_rate_{:.2f}_targeted_{}_w-patch-parameters".format(
                                                                                                train_location, 
                                                                                                train_location_min,
                                                                                                train_location_max,
                                                                                                train_alpha,
                                                                                                watermark_width,
                                                                                                poison_injection_rate,
                                                                                                targeted)))
                else:
                    poisoned_file = file.split()[0].replace(os.path.join(source_path, 'train'), 
                                                                    os.path.join(poisoned_destination_path, 
                                                                    "train",
                                                                    "loc_{}_loc-min_{:.2f}_loc-max_{:.2f}_alpha_{:.2f}_width_{}_rate_{:.2f}_targeted_{}".format(
                                                                                                train_location, 
                                                                                                train_location_min,
                                                                                                train_location_max,
                                                                                                train_alpha,
                                                                                                watermark_width,
                                                                                                poison_injection_rate,
                                                                                                targeted)))
                poisoned_image.save(poisoned_file)
                if patch_params:
                    f_train.write(poisoned_file + " " + file.split()[1] + " " + str(location[0]) + " " + str(location[1]) + " " + str(w_width) + " " + str(w_height) + "\n")
                else:
                    f_train.write(poisoned_file + " " + file.split()[1] + "\n")
            else:                      
                f_train.write(file + "\n")

    # close files
    for split in splits:
        if split == "train":
            f_train.close()
        if split == "val_poisoned":
            f_val_poisoned.close()
    logging.info("Finished creating ImageNet poisoned subset at {}!".format(poisoned_destination_path))


if __name__ == '__main__':
    main()
