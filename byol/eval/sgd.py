import torch
import torch.nn as nn
import torch.optim as optim

import os
from tqdm import trange, tqdm

# Modified from original
def eval_sgd(x_train, y_train, x_test, y_test, x_test_p, y_test_p, clf_fname, chkpt=None, evaluate=False, topk=[1, 5], epoch=500):
    """ linear classifier accuracy (sgd) """

    if evaluate:
        output_size = x_test.shape[1]
        num_class = y_test.max().item() + 1
        clf = nn.Linear(output_size, num_class)
        clf.cuda()

        if chkpt: # resume clf 
            clf.load_state_dict(torch.load(chkpt))

        clf.eval()
        with torch.no_grad():
            y_pred = clf(x_test)
        pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
        acc = {
            t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
            for t in topk
        }


        with torch.no_grad():
            y_pred_p = clf(x_test_p)
        pred_top_p = y_pred_p.topk(max(topk), 1, largest=True, sorted=True).indices
        acc_p = {
            t: (pred_top_p[:, :t] == y_test_p[..., None]).float().sum(1).mean().cpu().item()
            for t in topk
        }
        return acc, y_pred, y_test, acc_p, y_pred_p, y_test_p


    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in tqdm(range(epoch)):
        # print("epoch {}".format(ep))
        perm = torch.randperm(len(x_train)).view(-1, 1000)
        for idx in perm:
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()

        if (ep + 1) % 50 == 0:
            os.makedirs(os.path.dirname(clf_fname), exist_ok=True)
            torch.save(clf.state_dict(), clf_fname.format(ep))                

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }
    
    del clf
    return acc
