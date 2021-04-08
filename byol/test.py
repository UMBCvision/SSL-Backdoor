import torch
from datasets import get_ds
from cfg import get_cfg
from methods import get_method

from eval.sgd import eval_sgd
from eval.knn import eval_knn
from eval.lbfgs import eval_lbfgs
from eval.get_data import get_data

import os
import numpy as np

if __name__ == "__main__":
    cfg = get_cfg()

    model_full = get_method(cfg.method)(cfg)
    model_full.cuda().eval()
    if cfg.fname is None:
        print("evaluating random model")
    else:
        model_full.load_state_dict(torch.load(cfg.fname))

    ds = get_ds(cfg.dataset)(None, cfg, cfg.num_workers)
    device = "cpu" if cfg.clf == "lbfgs" else "cuda"
    if cfg.eval_head:
        model = lambda x: model_full.head(model_full.model(x))
        out_size = cfg.emb
    else:
        model = model_full.model
        out_size = model_full.out_size

    if cfg.evaluate:
        x_train, y_train = "", ""
        x_test, y_test = get_data(model, ds.test, out_size, device)
        x_test_p, y_test_p = get_data(model, ds.test_p, out_size, device)
    else:
        x_train, y_train = get_data(model, ds.clf, out_size, device)
        x_test, y_test = get_data(model, ds.test, out_size, device)
        x_test_p, y_test_p = "", ""


    clf_fname = os.path.join(os.path.dirname(cfg.fname), "linear", os.path.basename(cfg.fname), "{}.pt")
    if cfg.clf == "sgd":
        if cfg.evaluate:
            acc, pred_var_stack, labels_var_stack, acc_p, pred_var_p_stack, labels_var_p_stack = eval_sgd(x_train, y_train, x_test, y_test, x_test_p, y_test_p, clf_fname, chkpt=cfg.clf_chkpt, evaluate=cfg.evaluate)    


            pred_var_stack = torch.argmax(pred_var_stack, dim=1)
            pred_var_p_stack = torch.argmax(pred_var_p_stack, dim=1)

            # create confusion matrix ROWS ground truth COLUMNS pred
            conf_matrix_clean = np.zeros((int(labels_var_stack.max())+1, int(labels_var_stack.max())+1))
            conf_matrix_poisoned = np.zeros((int(labels_var_stack.max())+1, int(labels_var_stack.max())+1))

            for i in range(pred_var_stack.size(0)):
                # update confusion matrix
                conf_matrix_clean[int(labels_var_stack[i]), int(pred_var_stack[i])] += 1

            for i in range(pred_var_p_stack.size(0)):
                # update confusion matrix
                conf_matrix_poisoned[int(labels_var_p_stack[i]), int(pred_var_p_stack[i])] += 1

            # load imagenet metadata
            with open("imagenet_metadata.txt", "r") as f:
                data = [l.strip() for l in f.readlines()]
                imagenet_metadata_dict = {}
                for line in data:
                    wnid, classname = line.split('\t')[0], line.split('\t')[1]
                    imagenet_metadata_dict[wnid] = classname

            with open('imagenet100_classes.txt', 'r') as f:
                class_dir_list = [l.strip() for l in f.readlines()]
                class_dir_list = sorted(class_dir_list)

            save_folder = os.path.join(os.path.dirname(cfg.clf_chkpt), "linear", os.path.basename(cfg.clf_chkpt), cfg.eval_data)
            os.makedirs(save_folder, exist_ok=True)
            np.save("{}/conf_matrix_clean.npy".format(save_folder), conf_matrix_clean)
            np.save("{}/conf_matrix_poisoned.npy".format(save_folder), conf_matrix_poisoned)

            with open("{}/conf_matrix.csv".format(save_folder), "w") as f:
                f.write("Model {},,Clean val,,,,Pois. val,,\n".format(""))
                f.write("Data {},,acc1,,,,acc1,,\n".format(""))
                f.write(",,{:.2f},,,,{:.2f},,\n".format(acc[1]*100, acc_p[1]*100))
                f.write("class name,class id,TP,FP,,TP,FP\n")
                for target in range(len(class_dir_list)):
                    f.write("{},{},{},{},,".format(imagenet_metadata_dict[class_dir_list[target]].replace(",",";"), target, conf_matrix_clean[target][target], conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]))
                    f.write("{},{}\n".format(conf_matrix_poisoned[target][target], conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]))

        else:
            acc = eval_sgd(x_train, y_train, x_test, y_test, x_test_p, y_test_p, clf_fname)
    if cfg.clf == "knn":
        acc = eval_knn(x_train, y_train, x_test, y_test)
    elif cfg.clf == "lbfgs":
        acc = eval_lbfgs(x_train, y_train, x_test, y_test)
