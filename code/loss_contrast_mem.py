from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import  ramps

import logging

# from ..builder import LOSSES

# from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
#                                  cross_entropy, mask_cross_entropy)
# @LOSSES.register_module()
class PixelContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()
        self.temperature = 20
        self.base_temperature =0.1

        self.ignore_label = 255

        self.max_samples = 1024        #可能很重要，最大样本数
        self.max_views = 100       #可能很重要，最大视图数

    # def get_current_consistency_weight(self,epoch):
    #     return self.temperature* ramps.sigmoid_rampup(epoch, 40)
    # new_temperature = get_current_consistency_weight(epoch)

    def _hard_anchor_sampling(self, X, y_hat, y):             #强锚点采样  X（B，H*W，C） 输入一个批次的特征图和标签
        batch_size, feat_dim = X.shape[0], X.shape[-1]         #批次样本数B，特征维度C
        classes = []
        total_classes = 0           #整个批次中类别数量
        for ii in range(batch_size):        #从Y中找总类别数
            this_y = y_hat[ii]         #当前图像的真实标签
            this_classes = torch.unique(this_y)       #将该图像所有得类别返回（1，2，3，4……）,提取不重复的元素
            this_classes = [x for x in this_classes if x != self.ignore_label]          #剔除空背景，剔除像素太少的样本（maxview）
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0] > self.max_views]

            classes.append(this_classes)            #每个样本（图像）中的有效class
            total_classes += len(this_classes)           #总类别

        n_view =100               #防止某个类别的样本数超过预设的最大视图数，每类像素不超过100

        X_ = torch.zeros((2, n_view, feat_dim), dtype=torch.float).cuda()       #存放每个类的像素（视图）
        y_ = torch.tensor([0, 1])

        X_ptr = 0             #指针，用于追踪储存特征的张量X_中的当前位置
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]      #当前图像ii的真实标签
            this_y = y[ii]              #当前图像ii的预测标签
            cls_id = 1
            hard_indices_out = ((this_y_hat == 0) & (this_y == 0)).nonzero(as_tuple=False)   #P中不为Y的部分（歧义）
            # hard_indices_in = ((this_y_hat == 0) & (this_y == 0)).nonzero(as_tuple=False)   #P中不为Y的部分（歧义）
            # indices_tensor1 = hard_indices_out[torch.randperm(hard_indices_out.size(0))[:30]]
            # indices_tensor2 = hard_indices_in[torch.randperm(hard_indices_in.size(0))[:70]]
            # hard_indices = torch.cat((indices_tensor1,indices_tensor2), dim=0)
            # hard_indices = hard_indices[torch.randperm(hard_indices.size(0))]
            hard_indices = hard_indices_out

            easy_indices_out = ((this_y_hat == 1) & (this_y == 1)).nonzero(as_tuple=False)
            easy_indices_in = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)   #P和Y重叠部分（认可）
            indices_tensor1 = easy_indices_out[torch.randperm(easy_indices_out.size(0))[:60]]
            indices_tensor2 = easy_indices_in[torch.randperm(easy_indices_in.size(0))[:40]]
            easy_indices = torch.cat((indices_tensor1,indices_tensor2), dim=0)
            # hard_indices = hard_indices[torch.randperm(hard_indices.size(0))]
            # easy_indices = torch.cat((easy_indices_in, easy_indices_out), dim=0)
            # easy_indices = easy_indices_out
            #硬索引：预测标签与类别一样且类别与当前索引不一样的像素位置，nonzero表示返回非零元素索引，as_tuple表示返回二维张量
            in_indices = (this_y_hat == 1).nonzero(as_tuple=False)
            out_indices = (this_y_hat == 0).nonzero(as_tuple=False)
            num_hard = hard_indices.shape[0]
            num_easy = easy_indices.shape[0]
            print(num_hard,"and",num_easy)

            if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view // 2         #主要保留hard

            elif num_easy <= n_view / 2:
                    easy_indices =  torch.cat((easy_indices,in_indices[:n_view-num_easy]),dim=0)
                    num_easy_keep = n_view//2
                    num_hard_keep= n_view//2
            # else:
            #     logging.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
            #     raise Exception
            if num_hard <= n_view / 2:
                    hard_indices = torch.cat((hard_indices, in_indices[:n_view - num_hard]), dim=0)
                    num_easy_keep = n_view // 2
                    num_hard_keep = n_view // 2
            if num_easy ==0:
                    easy_indices = hard_indices

            perm = torch.randperm(num_hard_keep)          #生成随机排列整数序列
            hard_indices = hard_indices[perm[:num_hard_keep]]          #从中打乱选取前“num_hard_keep”个索引
            perm = torch.randperm(num_easy_keep)
            easy_indices = easy_indices[perm[:num_easy_keep]]          #从中打乱选取前“num_easy_keep”个索引
            # indices = torch.cat((hard_indices,easy_indices),dim=0)
            indices_keep = 50
            X_[ii, 0:50, :] = X[ii, easy_indices, :].squeeze(1)
            X_[ii, 50:100, :] = X[ii, hard_indices, :].squeeze(1)
            # indices_keep = 50
            # X_[0, indices_keep*ii:indices_keep*ii+50, :] =  X[ii, easy_indices, :].squeeze(1)
            # X_[1, indices_keep*ii:indices_keep*ii+50, :] = X[ii, hard_indices, :].squeeze(1)

            X_ptr += 1          #batch_size

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]
        y_anchor = y_anchor.contiguous().view(-1, 1)         #将y_anchor转换为二维张量4，1
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)  #沿第二维度拆开，沿第一维度拼接。最后为（200，2）


        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        # 定义全 1 和全 0 块
        small_tensor = torch.tensor([[1, 0], [0, 1]])
        block_1 = torch.ones(100, 100)
        mask = torch.kron(small_tensor, block_1).float().cuda()  # （200，200）
        anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
        contrast_feature = F.normalize(anchor_feature, p=2, dim=1)
        logits = torch.matmul(anchor_feature, contrast_feature.T)/self.base_temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)  # （400，1）
        logits = logits - logits_max.detach()  # 400个样本所有做矩阵乘法
        neg_mask = 1 - mask
        logits_mask = torch.ones_like(mask).scatter_(1,  # 对角线赋0了
                                                     torch.arange(anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask  # logits_mask为一个（400，400）对角线为0，其余全1的mask
        # a = (mask * logits).sum(0)
        # b = (neg_mask * logits).sum(0)
        # result = torch.where(a > b, torch.tensor(1.0).to('cuda'), torch.tensor(0.0).to('cuda'))
        # # print(result)
        # num_ones = torch.sum(result == 1.0).item()
        # print("正样本比负样本对大的个数", num_ones)

        neg_logits = torch.exp(logits) * neg_mask  # 计算负样本对的logits
        neg_logits = neg_logits.sum(1, keepdim=True)  # 两个样本同类的体素特征logits，再沿着第二个维度求和，每个体素的logits
        # 一个体素，对所有体素（负样本）的相似度之和
        # print(neg_logits.view(1,100))
        exp_logits = torch.exp(logits)
        # log_prob = logits - torch.log(exp_logits + neg_logits)  # 每个正样本对+所有负样本的对数概
        # # mean_log_prob_pos = ((mask * log_prob).sum(1)) / mask.sum(1)
        # print(exp_logits.sum(1))
        fenzi = exp_logits
        fenmu = (exp_logits + neg_logits)
        log_prob_pos = torch.log((fenzi / fenmu))
        mean_log_prob_pos = ((log_prob_pos * mask).sum(1)) / mask.sum(1)

        loss= -(mean_log_prob_pos / self.temperature)  # 200个
        loss = loss.mean()
        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        batch_size = feats.shape[0]
        labels = labels.contiguous().view(batch_size, -1)              #将标签展成二维张量B，H*W
        predict = predict.contiguous().view(batch_size, -1)            #-1是一个特俗标记，表示维度重新计算为了匹配张量总元素数量
        feat_adj = feats.permute(0, 2, 3, 4, 1)
        feat_adj = feat_adj.contiguous().view(feats.shape[0], -1, feat_adj.shape[-1])  #将特征展平成三维张量（B，H*W，C）
        feats_, labels_ = self._hard_anchor_sampling(feat_adj, labels, predict)

        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss

# @LOSSES.register_module()
class ContrastCELoss(nn.Module, ABC):
    def __init__(self, loss_weight=1.0):
        super(ContrastCELoss, self).__init__()

        # logging.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = loss_weight
        self.use_rmi = False
        self.use_lovasz = False
        self.seg_criterion = CrossEntropyLoss()
        self.contrast_criterion = PixelContrastLoss()

    def forward(self, seg, target, embed, with_embed=False):
        h, w = target.size(1), target.size(2)
        # test
        preds = {}  #区域样本和像素样本互为补充
        if "segment_queue" in preds:
            segment_queue = preds['segment_queue']
        else:
            segment_queue = None

        if "pixel_queue" in preds:
            pixel_queue = preds['pixel_queue']
        else:
            pixel_queue = None

        # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(seg, target)

        if segment_queue is not None and pixel_queue is not None:
            queue = torch.cat((segment_queue, pixel_queue), dim=1)

            _, predict = torch.max(seg, 1)
            loss_contrast = self.contrast_criterion(embed, target, predict, queue)
        else:
            _, predict = torch.max(seg, 1)
            loss_contrast = self.contrast_criterion(embed, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast
        # print(loss_contrast)
        return loss  + 1 * loss_contrast # just a trick to avoid errors in distributed training
