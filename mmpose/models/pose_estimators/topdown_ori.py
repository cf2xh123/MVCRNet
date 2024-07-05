# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from itertools import zip_longest, product
from typing import Optional

import cv2
import torch
from matplotlib import pyplot as plt
from torch import Tensor
import torchvision.transforms as transforms
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator
import numpy as np
from torch.nn import functional as F

from ...datasets.transforms.formatting import keypoints_to_tensor
from ...utils.tensortrans import total_tensortrans_w


@MODELS.register_module()
class TopdownPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
        self.new_kpt = {}
        self.new_kpt_w = {}

    def loss(self, inputs: Tensor, data_samples: SampleList,epoch:int =None,ema_model=None,re_model=None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """

        #input_ori = inputs.clone()
        #self.train_cfg['inputs'] = inputs.clone()
        #inputs_flip,use_fli = self.tensor_flip(inputs.clone())
        #self.train_cfg['flip'] = use_fli
        #self.train_cfg['inputs_flip'] = inputs_flip.clone()

        '''
        for ba in range(inputs.shape[0]):
         flip = inputs_flip[ba].cpu().numpy().transpose(1,2,0)

         cv2.imshow('flip'+str(ba),flip)
        self.train_cfg['flip'] = use_fli
        cv2.waitKey()'''

        #re_tager_model = copy.deepcopy(re_model)
        inputs_ori = inputs.clone()


        feats = self.extract_feat(inputs)


        losses = dict()
        '''
        inputs_1 = inputs.clone()
        inputs_2 = inputs.clone()
        out1,out2= self.resample(inputs_1,inputs_2,data_samples,epoch)
        weights_re = self.get_rekpt(out1,out2)'''
        if epoch != None:
            self.train_cfg['epoch'] = epoch

            #self.train_cfg['weights_re'] = weights_re
            #self.train_cfg['out_re'] = out2+

        '''
        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in data_samples
        ],
                         dim=0)
        gt_simcc = (gt_x, gt_y)
        gt_simcc  = self.head.decode(gt_simcc)
        for n in range(0,inputs.shape[0]):
            img = inputs[n].cpu().numpy().transpose(1,2,0)
            use_lable = data_samples[n].gt_instances['use_label'][0]
            kpt = gt_simcc[n]['keypoints'][0]
            color = (0, 255, 0)  # 绿色
            radius = 2
            image_with_keypoints = img.copy()
            for point in kpt:
                cv2.circle(image_with_keypoints, (int(point[0]),int(point[1])), radius, color, -1)
            filename = 'data/ap10k/imgs/re/' + str(data_samples[n].raw_ann_info['image_id'])+'_'+str(epoch)+'_real.jpg'
            cv2.imwrite(filename,image_with_keypoints*255)
            '''


        if self.with_head:
            loss_sm,pred_x,pred_y,keypoint_weights,loss_weight,gt= self.head.loss(feats, data_samples, train_cfg=self.train_cfg)
            #loss_sm,pred_x,pred_y,keypoint_weights,target_final= self.head.loss(feats, data_samples, train_cfg=self.train_cfg)
            '''
            target_x,target_y ,target_w = target_final
            target_w = target_w.reshape((inputs.shape[0],17)).cpu().numpy()
            target_out = self.head.decode((target_x,target_y))

            for n in range(0, inputs.shape[0]):
                img = inputs[n].cpu().numpy().transpose(1, 2, 0)
                weights = target_w[n]
                kpt = target_out[n]['keypoints'][0]
                color = (0, 255, 0)  # 绿色
                radius = 2
                image_with_keypoints = img.copy()
                kpt_n =0
                for point in kpt:
                    if weights[kpt_n] != 0:
                      cv2.circle(image_with_keypoints, (int(point[0]), int(point[1])), radius, color, -1)
                    kpt_n = kpt_n + 1
                filename = 'data/ap10k/imgs/re/' + str(data_samples[n].raw_ann_info['image_id']) + '_' + str(
                    epoch) + '_re.jpg'
                cv2.imwrite(filename, image_with_keypoints*255)'''

            out_ori = (pred_x,pred_y)
            batch_shape = pred_x.shape[0]
            kpt_shape = pred_x.shape[1]

        inputs_ema = torch.cat([
                d.img_w.unsqueeze(0).cuda().clone() for d in data_samples
            ])



        if ema_model != None:
            #print(111111)


            out_ema = ema_model(inputs_ema.clone(), data_samples)

            ema_losses = self.ema_loss(out_ori=out_ori,out_ema=out_ema,weight=keypoint_weights)
            ema_rate = self.get_current_consistency_weight(const_weight=2.0,epoch=epoch-210,consistency_rampup=10)





            ema_losses = ema_losses *ema_rate

            loss_all = {#'loss_kpt' : loss_sm['loss_kpt']  ,
                        #'loss_kt' : loss_sm['loss_kpt']  ,
                        'loss_ema' : ema_losses,
                        'acc_pose': loss_sm['acc_pose'],
                        }
        else:
            loss_all = loss_sm

        re_epoch = 210

        if epoch >= re_epoch:
          with torch.no_grad():

            inputs_w = torch.cat([
                d.img_w.unsqueeze(0).cuda().clone() for d in data_samples
            ])#.clone
            batch_shape = inputs.shape[0]
            kpt_shape = 17

            input_ema = inputs_w.clone()

            from mmpose.utils.tensortrans import total_tensortrans
            loss_weight_1 = loss_weight[0] + loss_weight[1]
            loss_weight_1 = loss_weight_1.clone().reshape((batch_shape, kpt_shape))  # 小损失
            loss_weight_1[loss_weight_1 > 0] = 1
            loss_weight_re = torch.ones_like(loss_weight_1) - loss_weight_1


            tensortrans = total_tensortrans()
            img_trans = inputs_w.clone()

            input_trans, use_flip_treans = self.tensor_flip(img_trans.clone(),0.5)
            input_trans,r_trans = self.tensor_rat(input_trans.clone(),80,0.5)
            input_trans ,scale =  self.input_scale(input_trans.clone(),0.5,1.5)
            input_trans = tensortrans(input_trans.clone())
            input_trans_res = re_model._forward(input_trans)

            kpt_trans = self.head.decode(input_trans_res)
            W, H = inputs_w.shape[2], inputs_w.shape[3]
            for bat_nu in range(len(kpt_trans)):
                '''
                img_rt = input_trans[bat_nu].cpu().numpy().transpose(1, 2, 0)
                color = (0, 255, 0)  # 绿色
                radius = 5
                image_with_keypoints = img_rt.copy()
                for kpt_bat in range(kpt_trans[bat_nu]['keypoints'].shape[1]):
                    if kpt_trans[bat_nu]['keypoint_scores'][0][kpt_bat]!= 0:
                      x ,y = kpt_trans[bat_nu]['keypoints'][0][kpt_bat]
                      cv2.circle(image_with_keypoints, (int(x), int(y)), radius, color, -1)'''
                if r_trans[bat_nu] != 0:
                    kpt_trans[bat_nu] = self.rotate_keypoints(kpt_trans[bat_nu], r_trans[bat_nu], W, H)
                if use_flip_treans[bat_nu] == 1:
                    flip_indices = data_samples[bat_nu].flip_indices
                    kpt_trans[bat_nu]['keypoints'] = kpt_trans[bat_nu]['keypoints'].take(flip_indices, axis=1)
                    kpt_trans[bat_nu]['keypoint_scores'] = kpt_trans[bat_nu]['keypoint_scores'].take(flip_indices, axis=1)
                    kpt_trans[bat_nu]['keypoints'][..., 0] = W - 1 - kpt_trans[bat_nu]['keypoints'][..., 0]
                kpt_trans[bat_nu] = self.kpt_scale(kpt_trans[bat_nu], scale[bat_nu],W)
                '''
                img = inputs_w[bat_nu].cpu().numpy().transpose(1, 2, 0)
                color = (0, 255, 0)  # 绿色
                radius = 5
                image_with_keypoints_2 = img.copy()
                for kpt_bat in range(kpt_trans[bat_nu]['keypoints'].shape[1]):
                    if kpt_trans[bat_nu]['keypoint_scores'][0][kpt_bat]!= 0:
                      x ,y = kpt_trans[bat_nu]['keypoints'][0][kpt_bat]
                      cv2.circle(image_with_keypoints_2, (int(x), int(y)), radius, color, -1)
                imgs = np.hstack([image_with_keypoints, image_with_keypoints_2])
                cv2.imshow('trams_'+str(bat_nu),imgs)'''

            del input_trans,use_flip_treans, img_trans,input_trans_res


            img_re = inputs_w.clone()
            tensortrans_w = total_tensortrans_w()

            img_re, use_flip_re = self.tensor_flip(img_re.clone(),0.5)
            img_re,r_re = self.tensor_rat(img_re.clone(),40,0.5)
            img_re ,scale_re =  self.input_scale(img_re.clone(),0.75,1.25)
            img_re = tensortrans_w(img_re.clone())
            out_re = re_model._forward(img_re.clone())
            kpt_re = self.head.decode(out_re)
            W, H = inputs_w.shape[2], inputs_w.shape[3]
            for bat_nu in range(len(kpt_re)):
                '''
                img = inputs_w[bat_nu].cpu().numpy().transpose(1, 2, 0)
                color = (0, 255, 0)  # 绿色
                radius = 5
                image_with_keypoints_2 = img.copy()
                img_re_ori = img_re[bat_nu].cpu().numpy().transpose(1, 2, 0)
                color = (0, 255, 0)  # 绿色
                radius = 5
                image_with_keypoints = img_re_ori.copy()
                for kpt_bat in range(kpt_re[bat_nu]['keypoints'].shape[1]):
                    if kpt_re[bat_nu]['keypoint_scores'][0][kpt_bat]!= 0:
                      x ,y = kpt_re[bat_nu]['keypoints'][0][kpt_bat]
                      cv2.circle(image_with_keypoints, (int(x), int(y)), radius, color, -1)'''

                if r_re[bat_nu] != 0:
                    kpt_re[bat_nu] = self.rotate_keypoints(kpt_re[bat_nu], r_re[bat_nu], W, H)
                if use_flip_re[bat_nu] == 1:
                    flip_indices = data_samples[bat_nu].flip_indices
                    kpt_re[bat_nu]['keypoints'] = kpt_re[bat_nu]['keypoints'].take(flip_indices, axis=1)
                    kpt_re[bat_nu]['keypoint_scores'] = kpt_re[bat_nu]['keypoint_scores'].take(flip_indices, axis=1)
                    kpt_re[bat_nu]['keypoints'][..., 0] = W - 1 - kpt_re[bat_nu]['keypoints'][..., 0]
                kpt_re[bat_nu] = self.kpt_scale(kpt_re[bat_nu], scale_re[bat_nu],W)
                '''
                for kpt_bat in range(kpt_re[bat_nu]['keypoints'].shape[1]):
                    if kpt_re[bat_nu]['keypoint_scores'][0][kpt_bat]!= 0:
                      x, y = kpt_re[bat_nu]['keypoints'][0][kpt_bat]
                      cv2.circle(image_with_keypoints_2, (int(x), int(y)), radius, color, -1)
                imgs = np.hstack([image_with_keypoints, image_with_keypoints_2])
                cv2.imshow('re_' + str(bat_nu), imgs)'''


            del out_re,use_flip_re,img_re,r_re



            input_ema, use_flip_ema = self.tensor_flip(input_ema.clone(),0.5)
            input_ema,r_ema = self.tensor_rat(input_ema.clone(),40)
            ema_re_model = copy.deepcopy(ema_model).eval()
            out_ema_re = ema_re_model(input_ema, data_samples)
            kpt_ema_re = self.head.decode(out_ema_re)

            W, H = inputs_w.shape[2], inputs_w.shape[3]
            for bat_nu in range(len(kpt_ema_re)):
                '''
                img_ema_ori = input_ema[bat_nu].cpu().numpy().transpose(1, 2, 0)
                color = (0, 255, 0)  # 绿色
                radius = 5
                image_with_keypoints = img_ema_ori.copy()
                for kpt_bat in range(kpt_ema_re[bat_nu]['keypoints'].shape[1]):
                    x ,y = kpt_ema_re[bat_nu]['keypoints'][0][kpt_bat]
                    cv2.circle(image_with_keypoints, (int(x), int(y)), radius, color, -1)'''
                if r_ema[bat_nu] != 0:
                    kpt_ema_re[bat_nu] = self.rotate_keypoints(kpt_ema_re[bat_nu], r_ema[bat_nu], W, H)
                if use_flip_ema[bat_nu] == 1:
                    flip_indices = data_samples[bat_nu].flip_indices
                    kpt_ema_re[bat_nu]['keypoints'] = kpt_ema_re[bat_nu]['keypoints'].take(flip_indices, axis=1)
                    kpt_ema_re[bat_nu]['keypoint_scores'] = kpt_ema_re[bat_nu]['keypoint_scores'].take(flip_indices, axis=1)
                    kpt_ema_re[bat_nu]['keypoints'][..., 0] = W - 1 - kpt_ema_re[bat_nu]['keypoints'][..., 0]
                '''
                img = inputs_w[bat_nu].cpu().numpy().transpose(1, 2, 0)
                color = (0, 255, 0)  # 绿色
                radius = 5
                image_with_keypoints_2 = img.copy()
                for kpt_bat in range(kpt_ema_re[bat_nu]['keypoints'].shape[1]):
                    if kpt_ema_re[bat_nu]['keypoint_scores'][0][kpt_bat]!= 0:
                      x ,y = kpt_ema_re[bat_nu]['keypoints'][0][kpt_bat]
                      cv2.circle(image_with_keypoints_2, (int(x), int(y)), radius, color, -1)
                imgs = np.hstack([image_with_keypoints, image_with_keypoints_2])
                cv2.imshow('ema_'+str(bat_nu),imgs)'''


            del out_ema_re,use_flip_ema,input_ema
            #cv2.waitKey()




            kpt_re_w = np.zeros([batch_shape, kpt_shape])
            gt_x = torch.cat([
                d.gt_instance_labels.keypoint_x_labels.clone() for d in data_samples
            ],
                dim=0)
            gt_y = torch.cat([
                d.gt_instance_labels.keypoint_y_labels.clone() for d in data_samples
            ],
                dim=0)
            gt_simcc = (gt_x, gt_y)
            kpt_gt = self.head.decode(gt_simcc)

            for bat_size in range(batch_shape):
                # print(data_samples[bat_size].id)
                for kpt_size in range(kpt_shape):
                    if loss_weight_re[bat_size][kpt_size] == 1:

                        x1 = kpt_trans[bat_size]['keypoints'][0][kpt_size][0]
                        y1 = kpt_trans[bat_size]['keypoints'][0][kpt_size][1]
                        x2 = kpt_re[bat_size]['keypoints'][0][kpt_size][0]
                        y2 = kpt_re[bat_size]['keypoints'][0][kpt_size][1]
                        x3 = kpt_ema_re[bat_size]['keypoints'][0][kpt_size][0]
                        y3 = kpt_ema_re[bat_size]['keypoints'][0][kpt_size][1]
                        d1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                        d2 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
                        d3 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
                        d = max(d1,d2,d3)

                        #print(conf_ori,conf_re,conf_trans,conf_ema_re,conf)

                        # print(data_samples[bat_size].img_id)
                        if kpt_re[bat_size]['keypoint_scores'][0][kpt_size] > 0.8 and \
                                kpt_trans[bat_size]['keypoint_scores'][0][kpt_size] >0.8 and kpt_ema_re[bat_size]['keypoint_scores'][0][kpt_size] >0.9:
                            conf_re = kpt_re[bat_size]['keypoint_scores'][0][kpt_size]
                            conf_trans = kpt_trans[bat_size]['keypoint_scores'][0][kpt_size]
                            conf_ema_re = kpt_ema_re[bat_size]['keypoint_scores'][0][kpt_size]
                            if conf_re >= conf_trans and conf_re >= conf_ema_re:
                                new_x,new_y = x2,y2
                            elif conf_trans >= conf_re and conf_trans >= conf_ema_re:
                                new_x,new_y = x1,y1
                            elif conf_ema_re >= conf_trans and conf_ema_re >= conf_re:
                                new_x,new_y = x3,y3
                            '''
                            img = inputs_w[bat_nu].cpu().numpy().transpose(1, 2, 0)
                            color = (0, 255, 0)  # 绿色
                            color_2 = (255, 255, 0)  # 绿色
                            color_3 = (0, 255, 255)  # 绿色
                            color_4 = (0, 0, 255)  # 绿色
                            radius = 5
                            image_with_keypoints = img.copy()
                            image_with_keypoints_2 = img.copy()
                            image_with_keypoints_3 = img.copy()
                            image_with_keypoints_4 = img.copy()
                            cv2.circle(image_with_keypoints, (int(x1), int(y1)), radius, color, -1)
                            cv2.circle(image_with_keypoints_2, (int(x2), int(y2)), radius, color_2, -1)
                            cv2.circle(image_with_keypoints_3, (int(x3), int(y3)), radius, color_3, -1)
                            cv2.circle(image_with_keypoints_4, (int(new_x), int(new_y)), radius, color_4, -1)
                            imgs = np.hstack([image_with_keypoints, image_with_keypoints_2, image_with_keypoints_3,image_with_keypoints_4])
                            cv2.imshow('all_1', imgs)
                            cv2.waitKey()'''
                            kpt_gt[bat_size]['keypoints'][0][kpt_size][0], \
                            kpt_gt[bat_size]['keypoints'][0][kpt_size][1] = new_x, new_y
                            kpt_re_w[bat_size][kpt_size] = 1
                        elif d < 6.4  and  kpt_re[bat_size]['keypoint_scores'][0][kpt_size] >= 0.4 and \
                                kpt_trans[bat_size]['keypoint_scores'][0][kpt_size] >= 0.4 and kpt_ema_re[bat_size]['keypoint_scores'][0][kpt_size] >= 0.4:
                            new_x = (x1 + x2 + x3) / 3
                            new_y = (y1 + y2 + y3) / 3
                            '''
                            img = inputs_w[bat_nu].cpu().numpy().transpose(1, 2, 0)
                            color = (0, 255, 0)  # 绿色
                            color_2 = (255, 255, 0)  # 绿色
                            color_3 = (0, 255, 255)  # 绿色
                            color_4 = (0, 0, 255)  # 绿色
                            radius = 5
                            image_with_keypoints = img.copy()
                            image_with_keypoints_2 = img.copy()
                            image_with_keypoints_3 = img.copy()
                            image_with_keypoints_4 = img.copy()
                            cv2.circle(image_with_keypoints, (int(x1), int(y1)), radius, color, -1)
                            cv2.circle(image_with_keypoints_2, (int(x2), int(y2)), radius, color_2, -1)
                            cv2.circle(image_with_keypoints_3, (int(x3), int(y3)), radius, color_3, -1)
                            cv2.circle(image_with_keypoints_4, (int(new_x), int(new_y)), radius, color_4, -1)
                            imgs = np.hstack([image_with_keypoints, image_with_keypoints_2, image_with_keypoints_3,image_with_keypoints_4])
                            cv2.imshow('all_2', imgs)
                            cv2.waitKey()'''
                            kpt_gt[bat_size]['keypoints'][0][kpt_size][0], \
                            kpt_gt[bat_size]['keypoints'][0][kpt_size][1] = new_x, new_y
                            kpt_re_w[bat_size][kpt_size] = 1
                    elif loss_weight_re[bat_size][kpt_size] == 0 and data_samples[bat_size].gt_instances['use_label'][0]==1 :
                        conf_re = kpt_re[bat_size]['keypoint_scores'][0][kpt_size]
                        conf_trans = kpt_trans[bat_size]['keypoint_scores'][0][kpt_size]
                        conf_ema_re = kpt_ema_re[bat_size]['keypoint_scores'][0][kpt_size]
                        conf_ori = data_samples[bat_size].gt_instances['keypoints_confident'][0][kpt_size]
                        min_conf = min(conf_re,conf_trans,conf_ema_re)



                        if min_conf >= conf_ori and conf_ori >= 0.15:
                            x1 = kpt_trans[bat_size]['keypoints'][0][kpt_size][0]
                            y1 = kpt_trans[bat_size]['keypoints'][0][kpt_size][1]
                            x2 = kpt_re[bat_size]['keypoints'][0][kpt_size][0]
                            y2 = kpt_re[bat_size]['keypoints'][0][kpt_size][1]
                            x3 = kpt_ema_re[bat_size]['keypoints'][0][kpt_size][0]
                            y3 = kpt_ema_re[bat_size]['keypoints'][0][kpt_size][1]
                            d1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                            d2 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
                            d3 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
                            d = max(d1, d2, d3)
                            if d <= 6.4:
                              new_x = (x1 + x2 + x3) / 3
                              new_y = (y1 + y2 + y3) / 3
                              ori_x, ori_y = kpt_gt[bat_size]['keypoints'][0][kpt_size][0], kpt_gt[bat_size]['keypoints'][0][kpt_size][1]
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][0], \
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][1] = new_x, new_y
                            elif  conf_ema_re>conf_re and conf_ema_re>conf_trans:
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][0], \
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][1] = x3, y3

                            elif conf_re > conf_ema_re and conf_re > conf_trans:
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][0], \
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][1] = x2, y2

                            elif conf_trans > conf_re and conf_trans > conf_ema_re:
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][0], \
                              kpt_gt[bat_size]['keypoints'][0][kpt_size][1] = x1, y1


                              '''
                              img = inputs_w[bat_nu].cpu().numpy().transpose(1, 2, 0)
                              color = (0, 255, 0)  # 绿色
                              color_2 = (255, 255, 0)  # 绿色
                              color_3 = (0, 255, 255)  # 绿色
                              color_4 = (0, 0, 255)  # 绿色
                              radius = 5
                              image_with_keypoints = img.copy()
                              image_with_keypoints_2 = img.copy()
                              image_with_keypoints_3 = img.copy()
                              image_with_keypoints_4 = img.copy()
                              image_with_keypoints_5 = img.copy()
                              cv2.circle(image_with_keypoints, (int(x1), int(y1)), radius, color, -1)
                              cv2.circle(image_with_keypoints_2, (int(x2), int(y2)), radius, color_2, -1)
                              cv2.circle(image_with_keypoints_3, (int(x3), int(y3)), radius, color_3, -1)
                              cv2.circle(image_with_keypoints_4, (int(new_x), int(new_y)), radius, color_4, -1)
                              cv2.circle(image_with_keypoints_5, (int(ori_x), int(ori_y)), radius, color_4, -1)
                              imgs = np.hstack([image_with_keypoints, image_with_keypoints_2, image_with_keypoints_3,
                                                image_with_keypoints_4,image_with_keypoints_5])
                              cv2.imshow('all_2', imgs)
                              cv2.waitKey()'''







          gt_kpt_x, gt_kpt_y, gt_kpt_w = self.emataget(kpt_gt)
          gt_kpt = (gt_kpt_x, gt_kpt_y)
          re_out = self._forward(inputs_ori)
          from mmpose.models.losses.classification_loss import KLDiscretLoss
          loss = KLDiscretLoss(use_target_weight=True, beta=10., label_softmax=True)
          kpt_re_w = keypoints_to_tensor(kpt_re_w).cuda()
          kpt_re_w = kpt_re_w.reshape(-1)
          w_x = loss_weight[0] + kpt_re_w
          w_y = loss_weight[1] + kpt_re_w
          re_w = (w_x,w_y)
          re_loss =loss(re_out, gt_kpt, re_w,re=1)
          #loss_all['re'] = re_loss
          loss_all['loss_kpt'] =  re_loss

        #from mmpose.models.losses.classification_loss import KLDiscretLoss
        #loss = KLDiscretLoss(use_target_weight=True, beta=10., label_softmax=True)
        #re_loss = loss(pred_out, gt, loss_weight, re=1)
        # loss_all['re'] = re_loss
        #loss_all['loss_re'] = re_loss



        losses.update(loss_all)
        return losses

    def input_scale(self,input1,min_scale,max_scale):

        batch_num = input1.shape[0]
        input1_w = input1.shape[2]
        input1_h = input1.shape[3]
        sacle = np.zeros(batch_num)
        for bat_n in range(batch_num):
            input_tensor = input1[bat_n]
            scale_now = np.random.uniform(min_scale,max_scale)
            sacle[bat_n] = scale_now
            w = int(input1_w * scale_now)
            h = int(input1_h * scale_now)
            resized_tensor = transforms.Resize((w,h))(input_tensor)

            if w < input1_w:
                padding = [int((input1_w -w)/2),int((input1_w -w)/2),int((input1_w -w)/2+(input1_w -w)%2),int((input1_w -w)/2+(input1_w -w)%2)]
                resized_tensor = transforms.Pad(padding)(resized_tensor)
                input1[bat_n] = resized_tensor
            elif w > input1_w:
                resized_tensor = transforms.CenterCrop((input1_w, input1_h))(resized_tensor)
                input1[bat_n] = resized_tensor

        return input1,sacle

    def kpt_scale(self,keypoints,scale,w):
        kpt = keypoints['keypoints']
        kpt_score= keypoints['keypoint_scores']
        bat = kpt.shape[0]
        mid = w / 2
        mid_ori = [mid,mid]

        ori_mid = [-mid,mid]
        for b in range(bat):
            if scale  < 1 and ((w -int(scale * w))%2==1 ):
                kpt[b][:,0] = abs((kpt[b][:,0] - mid + 0.5)/ scale + mid)
                kpt[b][:,1] = abs((mid - kpt[b][:,1] - 0.5)/ scale - mid)
            else:
                kpt[b][:,0] = abs((kpt[b][:,0] - mid )/ scale + mid)
                kpt[b][:,1] = abs((mid - kpt[b][:,1] )/ scale - mid)
            for kpt_n in range(kpt.shape[1]):
                if kpt[b][kpt_n][0] > (w - 1) or kpt[b][kpt_n][1] > (w - 1) :
                    kpt_score[b][kpt_n] = 0
            #keypoints_visible[b][keypoints[b]>(w-1)].
        keypoints['keypoints'] = kpt
        keypoints['keypoint_scores'] = kpt_score
        return keypoints

    def kpt_flip(self,keypoints,keypoints_visible,flip_indices,w):
        keypoints = keypoints.take(flip_indices, axis=1)
        keypoints_visible = keypoints_visible.take(flip_indices, axis=1)
        keypoints[..., 0] = w - 1 - keypoints[..., 0]
        return keypoints,keypoints_visible


    def emataget(self,kpts):
        from mmpose.datasets.transforms.formatting import keypoints_to_tensor
        re_out_x, re_out_y, re_out_w = [],[],[]
        for kpt in kpts:
            out_x,out_y= self.get_ematarger(kpt['keypoints'])
            re_out_x.append(out_x[0])
            re_out_y.append(out_y[0])
            re_out_w.append(kpt['keypoint_scores'][0])

        return keypoints_to_tensor(np.array(re_out_x)).cuda(),keypoints_to_tensor(np.array(re_out_y)).cuda(),keypoints_to_tensor(np.array(re_out_w)).cuda()
    def get_ematarger(self,keypoints):
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

        keypoints_split = self._map_coordinates( #
            keypoints,simcc_split_ratio)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 3-sigma rule
        radius = sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)

        for n, k in product(range(N), range(K)):

            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                continue

            mu_x, mu_y = mu

            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * sigma[0]**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * sigma[1]**2))



        return target_x, target_y

    def _map_coordinates(
        self,keypoints,simcc_split_ratio) :
        """Mapping keypoint coordinates into SimCC space."""

        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * simcc_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)


        return keypoints_split


    def get_current_consistency_weight(self,const_weight, epoch, consistency_rampup):
        return const_weight * self.sigmoid_rampup(epoch, consistency_rampup)

    def sigmoid_rampup(self,current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def ema_loss(self,out_ori,out_ema,weight):
        from mmpose.models.losses.classification_loss import KLDiscretLoss
        loss = KLDiscretLoss(use_target_weight=True,beta=10.,label_softmax=True)
        ema_loss = loss(out_ori,out_ema,weight)
        return ema_loss

    def tensor_flip(self,input1,p):
        batch_num = input1.shape[0]
        input1_w = input1.shape[2]
        use_flip = np.zeros(batch_num)
        for bat_n in range(batch_num):
            if  np.random.uniform(0,1) <= p:
                #print('flip')
                use_flip[bat_n] = 1
                #mg = input1[bat_n].cpu().numpy()
                #img = np.flip(img, axis=2)
                #input1[bat_n] = torch.from_numpy(img.copy()).cuda()
                input1[bat_n] = torch.flip(input1[bat_n],dims=[2])
                '''
                for n_shape in range(input1.shape[1]):
                    for x_shape in range(int(input1_w / 2)):
                        change = input1[bat_n][n_shape, :, x_shape].clone()
                        input1[bat_n][n_shape, :, x_shape] = input1[bat_n][n_shape, :, input1_w - x_shape - 1].clone()
                        input1[bat_n][n_shape, :, input1_w - x_shape - 1] = change'''
            #use_flip = np.zeros(batch_num)

        return input1,use_flip

    def tensor_rat(self,input,rot_angle,rot_p=0.5):
        B, C, H, W = input.shape
        rf = rot_angle
        r = np.zeros(B)
        for bat_num in range(B):
            if np.random.random() <= rot_p:
                angle = np.random.uniform(-rf, rf)#/180 * math.pi
                r[bat_num] = angle
                transform_matrix = torch.tensor([
                    [math.cos(angle), math.sin(-angle),0],
                    [math.sin(angle), math.cos(angle), 0]],device='cuda')

                #transform_matrix = transform_matrix.unsqueeze(0).repeat(C, 1, 1).unsqueeze(0)
                grid = F.affine_grid(transform_matrix.unsqueeze(0),  # 旋转变换矩阵
                                     (1,C, H, W))

                input[bat_num] = F.grid_sample(input[bat_num].unsqueeze(0),  # 输入tensor，shape为[B,C,W,H]
                                       grid,  # 上一步输出的gird,shape为[B,C,W,H]
                                       mode='nearest',align_corners=True)[0]  # 一些图像填充方法，这里我用的是最近邻



        return input,r

    def flip_back(self,input_all,img_shape):

        H, W = img_shape
        input = input_all['keypoints']
        for b_nu in range(input.shape[0]):
            for kpt_num in range(input.shape[1]):
                input[b_nu][kpt_num][0] = W - input[b_nu][kpt_num][0]


        input_all['keypoints'] = input
        return input_all

    import math

    def rotate_keypoints(self, keypoints_list, angle, image_width, image_height):
        kpt = keypoints_list['keypoints']
        kpt_score = keypoints_list['keypoint_scores']
        batch_number = kpt.shape[0]
        for bat_n in range(batch_number):
            center_x = image_width / 2.0
            center_y = image_height / 2.0

            for i in range(kpt.shape[1]):
                x, y = kpt[bat_n][i][0], kpt[bat_n][i][1]

                rotated_x = (x - center_x) *(math.cos(angle)) +(y - center_y) *(math.sin(-angle))
                rotated_y = (x - center_x) *(math.sin(angle)) + (y - center_y) *(math.cos(angle))
                rotated_x = rotated_x + center_x
                rotated_y = rotated_y + center_y

                if not (0 <= rotated_x < image_width and 0 <= rotated_y < image_height):
                    kpt_score[bat_n][i] = 0
                    kpt[bat_n][i] = [0, 0]
                else:
                    kpt[bat_n][i] = np.array([rotated_x, rotated_y])

        keypoints_list['keypoints'] = kpt
        keypoints_list['keypoint_scores'] = kpt_score

        return keypoints_list

        '''

            # 检查旋转后的点是否超出图像范围
            rotated_x, rotated_y = rotated_point
            if 0 <= rotated_x < image_width and 0 <= rotated_y < image_height:
                rotated_keypoints[i] = rotated_point
            else:
                # 超出范围的点坐标设置为0
                rotated_keypoints[i] = torch.tensor([0, 0])'''



    def resample(self,input1,input2,data_samples,epoch):
        input1 ,use_flip= self.tensor_flip(input1)
        input1 ,r =self.tensor_rat(input1)
        output1 =self._forward(input1)
        pred_1 = self.head.decode(output1)
        output2 =self._forward(input2)
        pred_2 = self.head.decode(output2)
        pred_2_ori = []
        batch_num = input1.shape[0]
        W,H =input1.shape[2], input1.shape[3]
        for bat_nu in range(batch_num):
            pred_2_ori.append(pred_2[bat_nu]['keypoints'].copy())
            if use_flip[bat_nu] == 1:
                kpt_flip = [[0, 1], [5, 8], [6, 9], [7, 10], [11, 14], [12, 15], [13, 16]]
                for flip in kpt_flip:
                    for nu in range(pred_1[bat_nu]['keypoints'].shape[0]):
                       change = pred_1[bat_nu]['keypoints'][nu][flip[0]][0].copy()
                       pred_1[bat_nu]['keypoints'][nu][flip[0]][0] = pred_1[bat_nu]['keypoints'][nu][flip[1]][0]
                       pred_1[bat_nu]['keypoints'][nu][flip[1]][0] = change
                pred_2[bat_nu] = self.flip_back(pred_2[bat_nu],(W,H))
            if r[bat_nu] != 0:
                pred_2[bat_nu] = self.rotate_keypoints(pred_2[bat_nu],r[bat_nu],W,H)
        '''
        for n in range(0,input1.shape[0]):
            img = input1[n].cpu().numpy().transpose(1,2,0)
            use_lable = data_samples[n].gt_instances['use_label'][0]
            kpt = pred_1[n]['keypoints'][0]
            color = (0, 255, 0)  # 绿色
            radius = 2
            image_with_keypoints = img.copy()
            for point in kpt:
                cv2.circle(image_with_keypoints, (int(point[0]),int(point[1])), radius, color, -1)
            filename = 'data/ap10k/imgs/pse/' + str(data_samples[n].raw_ann_info['image_id'])+'_'+str(epoch)+'_1.jpg'
            cv2.imwrite(filename,image_with_keypoints*255)
        for n in range(0,input1.shape[0]):
            img = input1[n].cpu().numpy().transpose(1,2,0)
            use_lable = data_samples[n].gt_instances['use_label'][0]
            kpt = pred_2[n]['keypoints'][0]
            color = (0, 255, 0)  # 绿色
            radius = 2
            image_with_keypoints = img.copy()
            for point in kpt:
                cv2.circle(image_with_keypoints, (int(point[0]),int(point[1])), radius, color, -1)
            filename = 'data/ap10k/imgs/pse/' + str(data_samples[n].raw_ann_info['image_id'])+'_'+str(epoch)+'_2.jpg'
            cv2.imwrite(filename,image_with_keypoints*255)
        for n in range(0,input1.shape[0]):
            img = input2[n].cpu().numpy().transpose(1,2,0)
            use_lable = data_samples[n].gt_instances['use_label'][0]
            kpt = pred_2_ori[n][0]
            color = (0, 255, 0)  # 绿色
            radius = 2
            image_with_keypoints = img.copy()
            for point in kpt:
                cv2.circle(image_with_keypoints, (int(point[0]),int(point[1])), radius, color, -1)
            filename = 'data/ap10k/imgs/pse/' + str(data_samples[n].raw_ann_info['image_id'])+'_'+str(epoch)+'_ori.jpg'
            cv2.imwrite(filename,image_with_keypoints*255)'''

        return pred_1 ,pred_2



    def get_rekpt(self,inputs1,inputs2):
        weights = []
        for b_num in range(len(inputs1)):
            input1 = inputs1[b_num]['keypoints']
            input2 = inputs2[b_num]['keypoints']
            dist = np.linalg.norm(input1-input2, axis=-1, keepdims=True)
            for c_n in range(inputs1[b_num]['keypoints'].shape[0]):
                input1 = inputs1[b_num]['keypoints'][c_n]
                input2 = inputs2[b_num]['keypoints'][c_n]
                for kpt_n in range(inputs1[b_num]['keypoints'].shape[1]):
                    in1 = input1[kpt_n]
                    in2 = input2[kpt_n]
                    if (in1[0] <= 0.5 or in1[0]>=255 or in1[1] <= 0.5 or in1[1]>=255):
                        dist[c_n][kpt_n][0] = 256
                    if (in2[0] <= 0.5 or in2[0]>=255 or in2[1] <= 0.5 or in2[1]>=255):
                        dist[c_n][kpt_n][0] = 256
            mask = (dist < 2.4)
            weights.append(mask.astype(float).reshape(-1))
        return np.array(weights)


    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
