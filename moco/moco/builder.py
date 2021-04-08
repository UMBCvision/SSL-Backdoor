# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoLosses(typing.NamedTuple):
    loss_contr: typing.Optional[torch.Tensor] = None
    logits_contr: typing.Optional[torch.Tensor] = None
    loss_align: typing.Optional[torch.Tensor] = None
    loss_unif: typing.Optional[torch.Tensor] = None

    def combine(self, contr_w: float=1, align_w: float=1, unif_w: float=1) -> torch.Tensor:
        assert not contr_w == align_w == unif_w == 0
        l = 0
        if contr_w != 0:
            assert self.loss_contr is not None
            l += contr_w * self.loss_contr
        if align_w != 0:
            assert self.loss_align is not None
            l += align_w * self.loss_align
        if unif_w != 0:
            assert self.loss_unif is not None
            l += unif_w * self.loss_unif
        return l


class MoCo(nn.Module):
    r"""
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999,
                 contr_tau=0.07, align_alpha=None, unif_t=None, unif_intra_batch=True, mlp=False):
        r"""
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m

        # l_contr
        self.contr_tau = contr_tau
        if contr_tau is not None:
            self.register_buffer('scalar_label', torch.zeros((), dtype=torch.long))
        else:
            self.register_parameter('scalar_label', None)

        # l_align
        self.align_alpha = align_alpha

        # l_unif
        self.unif_t = unif_t
        self.unif_intra_batch = unif_intra_batch

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        r"""
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        r"""
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        r"""
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        r"""
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            MoCoLosses object containing the loss terms (and logits if contrastive loss is used)
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        moco_loss_ctor_dict = {}

        # lazyily computed & cached!
        def get_q_bdot_k():
            if not hasattr(get_q_bdot_k, 'result'):
                get_q_bdot_k.result = (q * k).sum(dim=1)
            assert get_q_bdot_k.result._version == 0
            return get_q_bdot_k.result

        # lazyily computed & cached!
        def get_q_dot_queue():
            if not hasattr(get_q_dot_queue, 'result'):
                get_q_dot_queue.result = q @ self.queue.clone().detach()
            assert get_q_dot_queue.result._version == 0
            return get_q_dot_queue.result

        # l_contrastive
        if self.contr_tau is not None:
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = get_q_bdot_k().unsqueeze(-1)
            # negative logits: NxK
            l_neg = get_q_dot_queue()

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.contr_tau

            moco_loss_ctor_dict['logits_contr'] = logits
            moco_loss_ctor_dict['loss_contr'] = F.cross_entropy(logits, self.scalar_label.expand(logits.shape[0]))

        # l_align
        if self.align_alpha is not None:
            if self.align_alpha == 2:
                moco_loss_ctor_dict['loss_align'] = 2 - 2 * get_q_bdot_k().mean()
            elif self.align_alpha == 1:
                moco_loss_ctor_dict['loss_align'] = (q - k).norm(dim=1, p=2).mean()
            else:
                moco_loss_ctor_dict['loss_align'] = (2 - 2 * get_q_bdot_k()).pow(self.align_alpha / 2).mean()

        # l_uniform
        if self.unif_t is not None:
            sq_dists = (2 - 2 * get_q_dot_queue()).flatten()
            if self.unif_intra_batch:
                sq_dists = torch.cat([sq_dists, torch.pdist(q, p=2).pow(2)])
            moco_loss_ctor_dict['loss_unif'] = sq_dists.mul(-self.unif_t).exp().mean().log()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return MoCoLosses(**moco_loss_ctor_dict)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    r"""
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
