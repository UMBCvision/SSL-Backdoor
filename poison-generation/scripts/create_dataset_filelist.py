import os
import glob
import argparse

PERCENTAGE=1

#create filelist
def create_dataset_filelist(data_root, output_file_root):
    os.makedirs(output_file_root, exist_ok=True)
    # train
    with open(os.path.join(output_file_root, "train_ssl_{}_filelist.txt".format(PERCENTAGE)), "w") as f:
        dir_list = sorted(glob.glob(os.path.join(data_root, "train", "*")))
        for dir_index, dir in enumerate(dir_list):
            file_list = (glob.glob(os.path.join(dir, "*")))
            for file in file_list[:int(len(file_list)*PERCENTAGE)]:
                f.write(file+" "+str(dir_index)+"\n")


    # val
    with open(os.path.join(output_file_root, "val_ssl_filelist.txt"), "w") as f:
        dir_list = sorted(glob.glob(os.path.join(data_root, "val", "*")))
        for dir_index, dir in enumerate(dir_list):
            file_list = (glob.glob(os.path.join(dir, "*")))
            for file in file_list[:int(len(file_list)*PERCENTAGE)]:
                f.write(file+" "+str(dir_index)+"\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create dataset filelist")
    parser.add_argument("--data-root")
    parser.add_argument("--output-file-root")

    args = parser.parse_args()
    create_dataset_filelist(args.data_root, args.output_file_root)
