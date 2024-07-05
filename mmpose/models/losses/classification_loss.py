# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mmpose.registry import MODELS
from mmpose.datasets.transforms.formatting import keypoints_to_tensor

@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 reduction='mean',
                 use_sigmoid=False):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = F.binary_cross_entropy if use_sigmoid \
            else F.binary_cross_entropy_with_logits
        self.criterion = partial(criterion, reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            #print(output.shape, target.shape)
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = (loss * target_weight)
        else:
            loss = self.criterion(output, target)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
class JSDiscretLoss(nn.Module):
    """Discrete JS Divergence loss for DSNT with Gaussian Heatmap.

    Modified from `the official implementation
    <https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
    """

    def __init__(
        self,
        use_target_weight=True,
        size_average: bool = True,
    ):
        super(JSDiscretLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.size_average = size_average
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        """Kullback-Leibler Divergence."""

        eps = 1e-24
        kl_values = self.kl_loss((q + eps).log(), p)
        return kl_values

    def js(self, pred_hm, gt_hm):
        """Jensen-Shannon Divergence."""

        m = 0.5 * (pred_hm + gt_hm)
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def forward(self, pred_hm, gt_hm, target_weight=None):
        """Forward function.

        Args:
            pred_hm (torch.Tensor[N, K, H, W]): Predicted heatmaps.
            gt_hm (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        Returns:
            torch.Tensor: Loss value.
        """

        if self.use_target_weight:
            assert target_weight is not None
            assert pred_hm.ndim >= target_weight.ndim

            for i in range(pred_hm.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.js(pred_hm * target_weight, gt_hm * target_weight)
        else:
            loss = self.js(pred_hm, gt_hm)

        if self.size_average:
            loss /= len(gt_hm)

        return loss.sum()


@MODELS.register_module()
class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax.
        label_softmax (bool): Whether to use Softmax on labels.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
    """

    def __init__(self, beta=1.0, label_softmax=False, use_target_weight=True):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.beta , dim=1)
        eps = 1e-8
        loss = torch.mean(self.kl_loss(log_pt, labels +eps), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight,use_labels =None,re = None):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        num_joints = pred_simcc[0].size(1)
        loss = 0

        if self.use_target_weight :
            if re == None:
               weight = target_weight.reshape(-1)
        else:
            weight = 1.
        if re == None:
         for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            loss += self.criterion(pred, target).mul(weight).sum()
        else:
            for pred, target,weights in zip(pred_simcc, gt_simcc,target_weight):
                pred = pred.reshape(-1, pred.size(-1))
                target = target.reshape(-1, target.size(-1))

                loss += self.criterion(pred, target).mul(weights).sum()

        return loss / num_joints

@MODELS.register_module()
class Pre_loss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax.
        label_softmax (bool): Whether to use Softmax on labels.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
    """

    def __init__(self, beta=1.0, label_softmax=False, use_target_weight=True):
        super(Pre_loss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.use_target_weight = use_target_weight
        Ce_criterion = F.binary_cross_entropy
        Cel_criterion =  F.binary_cross_entropy_with_logits
        self.criterion_ce = partial(Ce_criterion, reduction='none')

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss

    def SCE_loss(self, dec_outs, labels):

        log_pt = F.softmax(dec_outs ,dim=1)
        #pt_np = log_pt.cpu().detach().numpy()
        labels = F.softmax(labels, dim=1)
        #lab_np = labels.cpu().detach().numpy()
        ce_loss = torch.mean(self.criterion_ce(log_pt, labels), dim=1)
        rce_loss = torch.mean(self.criterion_ce(labels, log_pt), dim=1)
        #sq_loss = nn.functional.cross_entropy(labels, log_pt)
        loss = 2*ce_loss + rce_loss
        #loss = 2 * ce_loss +   sq_loss

        return loss


    def cosine_rampdown(self,current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        # assert 0 <= current <= rampdown_length
        current = np.clip(current, 0.0, rampdown_length)
        return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

    def get_current_topkrate(self,epoch, rampdown_epoch, min_rate):
        r = self.cosine_rampdown(epoch, rampdown_epoch)
        return np.clip(r, min_rate, 1)

    def forward(self, pred_simcc, gt_simcc, target_weight,use_labels,train_cfg=None):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        num_batch = pred_simcc[0].size(0)
        num_joints = pred_simcc[0].size(1)
        loss = 0
        loss_all = 0
        if self.use_target_weight:

            weight = target_weight.reshape(-1)
            target_weight_real = torch.zeros_like(target_weight)
            for nu in range(num_batch):
                if use_labels[nu] == 0:
                    target_weight_real[nu] = target_weight[nu]
            weight_real = target_weight_real.reshape(-1)
        else:
            weight = 1.
        if 'out_re' in train_cfg:
            re_out_x,re_out_y,re_out_w = self.retaget(train_cfg['out_re'],train_cfg['weights_re'])
            re_out_x = keypoints_to_tensor(re_out_x)
            re_out_y = keypoints_to_tensor(re_out_y)
            re_simcc = (re_out_x,re_out_y)

        n = 0

        weights_small_x = torch.zeros_like(weight_real)
        weights_small_y = torch.zeros_like(weight_real)

        for pred, target in zip(pred_simcc, gt_simcc):
        #for pred, target,re_target in zip(pred_simcc, gt_simcc,re_simcc):
            pred_copy = pred.detach().clone()
            target_ori = target.detach().clone()
            pred = pred.reshape(-1, pred.size(-1))
            pred_copy = pred_copy.reshape(-1, pred_copy.size(-1))
            target_ori = target_ori.reshape(-1, target_ori.size(-1))
            num_visible_joints = torch.count_nonzero(target_weight)
            rate = self.get_current_topkrate(train_cfg['epoch'] - 210,30,0.8)
            num_small_loss_samples = int(num_visible_joints * rate)

            #loss = self.criterion(pred, target_re)
            loss_small =  self.criterion(pred_copy, target_ori)

            loss_bat = torch.reshape(loss_small,(num_batch,num_joints))
            loss_max = loss_small.max() * torch.ones_like(loss_bat)
            zero_weight = (target_weight > 0)
            loss_new = torch.where(zero_weight, loss_bat, loss_max)
            dim_last = loss_bat.size(-1)
            _, topk_idx = torch.topk(loss_new.flatten(), k=num_small_loss_samples , largest=False)
            topk_idx = topk_idx.unsqueeze(-1)
            small_loss_idx = torch.cat([topk_idx // dim_last, topk_idx % dim_last], dim=-1)
            weights_small_loss_re = torch.zeros_like(target_weight)
            weights_small_loss_re[small_loss_idx[:, 0], small_loss_idx[:, 1]] = 1  # 这样可以确定每一张图片中哪个点的loss最小，是最小就位1
            '''
            weights_re = train_cfg['weights_re'] * (1 - weights_small_loss_re.cpu().numpy())
            for b_n in range(weights_re.shape[0]):
                for k_n in range(weights_re.shape[1]):
                    if weights_re[b_n][k_n] == 1:
                        target[b_n][k_n] = re_target[b_n][k_n]
                        weights_small_loss_re[b_n][k_n] = weights_small_loss_re[b_n][k_n] + 1
            if n == 0 :
                target_out_x = target.detach().clone()
                n = n+1
            else:
                target_out_y = target.detach().clone()'''
            target = target.reshape(-1, target.size(-1))
            #loss_sm_re = self.SCE_loss(pred, target)
            loss_sm_re = self.criterion(pred, target)
            weights_small_loss_re =  weights_small_loss_re.reshape(-1)
            weight_all=  2 * weight_real + weights_small_loss_re

            if n == 0:
                weights_small_x = weight_all.clone()
            else:
                weights_small_y = weight_all.clone()

            loss_all +=  loss_sm_re.mul(weight_all).sum()
            n = n + 1

        return loss_all / num_joints,(weights_small_x,weights_small_y)#,(target_out_x,target_out_y,weight_all)

    def retaget(self,kpts,kpts_vis):
        re_out_x, re_out_y, re_out_w = [],[],[]
        for kpt,kpt_vis in zip(kpts,kpts_vis):
            out_x,out_y,out_weights= self.get_targer(kpt['keypoints'],np.expand_dims(kpt_vis,axis=0))
            re_out_x.append(out_x[0])
            re_out_y.append(out_y[0])
            re_out_w.append(out_weights[0])
        return np.array(re_out_x),np.array(re_out_y),np.array(re_out_w)
    def get_targer(self,keypoints,keypoints_visible):
        simcc_split_ratio = 2.0
        input_size = (256, 256)
        sigma = (5.66, 5.66)
        if isinstance(sigma, (float, int)):
            sigma = np.array([sigma, sigma])
        else:
            sigma = np.array(sigma)
        N, K, _ = keypoints.shape
        w, h = input_size
        W = np.around(w * simcc_split_ratio).astype(int)
        H = np.around(h * simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible,simcc_split_ratio)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 3-sigma rule
        radius = sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y = mu

            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * sigma[0]**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * sigma[1]**2))



        return target_x, target_y, keypoint_weights

    def _map_coordinates(
        self,keypoints,keypoints_visible,simcc_split_ratio) :
        """Mapping keypoint coordinates into SimCC space."""

        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * simcc_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)
        keypoint_weights = keypoints_visible.copy()

        return keypoints_split, keypoint_weights


@MODELS.register_module()
class InfoNCELoss(nn.Module):
    """InfoNCE loss for training a discriminative representation space with a
    contrastive manner.

    `Representation Learning with Contrastive Predictive Coding
    arXiv: <https://arxiv.org/abs/1611.05424>`_.

    Args:
        temperature (float, optional): The temperature to use in the softmax
            function. Higher temperatures lead to softer probability
            distributions. Defaults to 1.0.
        loss_weight (float, optional): The weight to apply to the loss.
            Defaults to 1.0.
    """

    def __init__(self, temperature: float = 1.0, loss_weight=1.0) -> None:
        super(InfoNCELoss, self).__init__()
        assert temperature > 0, f'the argument `temperature` must be ' \
                                f'positive, but got {temperature}'
        self.temp = temperature
        self.loss_weight = loss_weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes the InfoNCE loss.

        Args:
            features (Tensor): A tensor containing the feature
                representations of different samples.

        Returns:
            Tensor: A tensor of shape (1,) containing the InfoNCE loss.
        """
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp
        targets = torch.arange(n, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss * self.loss_weight
