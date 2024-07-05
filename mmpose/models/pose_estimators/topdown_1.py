# Copyright (c) OpenMMLab. All rights reserved.
import math
from itertools import zip_longest, product
from typing import Optional

import cv2
import torch
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator
import numpy as np
from torch.nn import functional as F

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

    def loss(self, inputs: Tensor, data_samples: SampleList,epoch:int =None,ema_model=None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """

        #input_ori = inputs.detach().clone()
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
            #self.train_cfg['out_re'] = out2

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
            loss_sm,pred_x,pred_y,keypoint_weights= self.head.loss(feats, data_samples, train_cfg=self.train_cfg)
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



        if ema_model != None:
            inputs_w = torch.cat([
                d.img_w.unsqueeze(0).cuda() for d in data_samples
            ])
            input_ema = inputs_w

            #input_ori, use_flip = self.tensor_flip(input_ori)
            #input_ori, r = self.tensor_rat(input_ori)
            #output_ori = self._forward(input_ori)

            out_ema = ema_model(input_ema, data_samples)
            #pred_ema = self.head.decode(out_ema)
            '''
            batch_num = input_ori.shape[0]
            W, H = input_ori.shape[2], input_ori.shape[3]
            for bat_nu in range(batch_num):
                if use_flip[bat_nu] == 1:
                    kpt_flip = [[0, 1], [5, 8], [6, 9], [7, 10], [11, 14], [12, 15], [13, 16]]
                    for flip in kpt_flip:
                        for nu in range(pred_ema[bat_nu]['keypoints'].shape[0]):
                            change = pred_ema[bat_nu]['keypoints'][nu][flip[0]][0].copy()
                            pred_ema[bat_nu]['keypoints'][nu][flip[0]][0] = pred_ema[bat_nu]['keypoints'][nu][flip[1]][0]
                            pred_ema[bat_nu]['keypoints'][nu][flip[1]][0] = change
                    pred_ema[bat_nu] = self.flip_back(pred_ema[bat_nu], (W, H))
                if r[bat_nu] != 0:
                    pred_ema[bat_nu] = self.rotate_keypoints(pred_ema[bat_nu], r[bat_nu], W, H)


            out_ema = self.emataget(pred_ema)'''

            ema_losses = self.ema_loss(out_ori=out_ori,out_ema=out_ema,weight=keypoint_weights)
            ema_rate = self.get_current_consistency_weight(const_weight=2.0,epoch=epoch-210,consistency_rampup=10)


            ema_losses = ema_losses *ema_rate

            loss_all = {'loss_kpt' : loss_sm['loss_kpt']  ,
                        'loss_ema' : ema_losses,
                        'acc_pose': loss_sm['acc_pose']
                        }
        else:
            loss_all = loss_sm
        losses.update(loss_all)


        return losses


    def emataget(self,kpts):
        from mmpose.datasets.transforms.formatting import keypoints_to_tensor
        re_out_x, re_out_y, re_out_w = [],[],[]
        for kpt in kpts:
            out_x,out_y= self.get_ematarger(kpt['keypoints'])
            re_out_x.append(out_x[0])
            re_out_y.append(out_y[0])

        return keypoints_to_tensor(np.array(re_out_x)).cuda(),keypoints_to_tensor(np.array(re_out_y)).cuda()
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

        keypoints_split = self._map_coordinates(
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

    def tensor_flip(self,input1):
        batch_num = input1.shape[0]
        input1_w = input1.shape[2]
        use_flip = np.zeros(batch_num)
        for bat_n in range(batch_num):
            if  np.random.uniform(0,1) >= 0.3:
                use_flip[bat_n] = 1
                for n_shape in range(input1.shape[1]):
                    for x_shape in range(int(input1_w / 2)):
                        change = input1[bat_n][n_shape, :, x_shape].clone()
                        input1[bat_n][n_shape, :, x_shape] = input1[bat_n][n_shape, :, input1_w - x_shape - 1].clone()
                        input1[bat_n][n_shape, :, input1_w - x_shape - 1] = change

        return input1,use_flip

    def tensor_rat(self,input):
        B, C, H, W = input.shape
        rf = 40
        r = np.zeros(B)
        for bat_num in range(B):
            if np.random.random() <= 0.8:
                angle = np.random.uniform(-rf * 2, rf * 2) * math.pi
                r[bat_num] = angle
                transform_matrix = torch.tensor([
                    [math.cos(angle), math.sin(-angle), 0],
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

    def rotate_keypoints(self,keypoints_list, angles_list, image_width, image_height):
        kpt = keypoints_list['keypoints']
        kpt_score = keypoints_list['keypoint_scores']
        batch_number = kpt.shape[0]
        center = image_width/2.00
        for bat_n in range(batch_number):
            for i in range(kpt.shape[1]):
                x, y = kpt[bat_n][i][0], kpt[bat_n][i][1]
                rotated_x = (x-center) * math.cos(-angles_list) - (y-center) * math.sin(-angles_list)
                rotated_y = (x-center) * math.sin(-angles_list) + (y-center) * math.cos(-angles_list)
                rotated_x = rotated_x +center
                rotated_y = rotated_y +center

                if (0 <= rotated_x < image_width and 0 <= rotated_y < image_height) == False:
                    kpt_score[bat_n][i] = 0
                    kpt[bat_n][i] = [0, 0]
                else:
                    kpt[bat_n][i] = np.array([rotated_x,rotated_y])
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
