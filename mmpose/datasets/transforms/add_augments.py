import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmpose.registry import TRANSFORMS
import  cv2
@TRANSFORMS.register_module()
class CowMask(BaseTransform):
    def __init__(self, method='mix', sigmas=None,random_rate = 1,cow_rate = 0.75):
        #self.crop_size = crop_size
        self.method = method
        self.random_rate = random_rate
        self.cow_rate =cow_rate
        if sigmas is None:
            self.sigmas = (13, 15, 17, 19, 21, 23, 25)
        else:
            self.sigmas = sigmas

    def transform(self, results: Dict) -> Optional[dict]:
        now_rate = np.random.rand()
        if now_rate > self.random_rate:
            return results
        img = results['img'].copy()
        cow_size = img.shape[:2]
        sigma = np.random.choice(self.sigmas)
        mask = self.generate_cow_mask(size=cow_size, sigma=sigma, method=self.method)
        mask = np.expand_dims(mask, axis=0)
        mask[mask == 0] = self.cow_rate
        for nu in range(0, 3):
            img[:, :, nu] = img[:, :, nu] * mask
        results['img'] = img

        '''
        if results['img_id'] == 7199:
            cv2.imshow(str(results['img_id']),img)
            cv2.waitKey(0)'''
        return results
    '''
    def __call__(self, labels):
        img = labels.get('img')
        cow_size = img.shape[:2]
        sigma = np.random.choice(self.sigmas)

        mask = self.generate_cow_mask(size=cow_size, sigma=sigma, method=self.method)

        mask = np.expand_dims(mask, axis=0)
        mask[mask == 0] = 0.6
        for nu in range(0, 3):
            img[:, :, nu] = img[:, :, nu] * mask
        labels['img'] = img
        return labels'''
    def generate_cow_mask(self,size, sigma, method):
            cow_mask = np.random.uniform(low=0.0, high=1.0, size=size)
            cow_mask_gauss = gaussian_filter(cow_mask, sigma=sigma)
            mean = np.mean(cow_mask_gauss)
            std = np.std(cow_mask_gauss)
            # thresh = mean + perturbation*std
            if method == "mix":
                cow_mask_final = (cow_mask_gauss < mean).astype(np.float64)
            elif method == "cut":
                offset = np.random.uniform(low=0.0, high=1.0, size=())
                cow_mask_final = (cow_mask_gauss < mean + offset * std).astype(np.float64)  # .astype(np.int32)
            else:
                raise NotImplementedError

            return cow_mask_final