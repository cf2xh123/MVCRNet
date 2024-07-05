import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import random
import math

class cowmask_tensor():

    def __init__(self, method='mix', sigmas=None):
            # self.crop_size = crop_size
            self.method = method
            if sigmas is None:
                self.sigmas = (13, 15, 17, 19, 21, 23, 25)
            else:
                self.sigmas = sigmas

    def generate_cow_mask(self, size, sigma, method):
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
    def __call__(self, tensor):
        #img = batchs
        batch_number ,cow_size = tensor.shape[0],tensor.shape[1:]
        sigma = np.random.choice(self.sigmas)

        mask = self.generate_cow_mask(size=cow_size, sigma=sigma, method=self.method)
        mask = np.array([[mask]*batch_number])
        mask = torch.Tensor(mask).cuda()
        #mask = np.expand_dims(mask, axis=0)
        mask[mask == 0] = 0.45
        tensor = tensor * mask
        #for nu in range(0, 3):
            #img[:, :, nu] = img[:, :, nu] * mask

        #batchs = img2
        return tensor


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h )
                y1 = random.randint(0, img.size()[2] - w )
                if img.size()[0] == 3:
                    # img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    # img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    # img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    #img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    #img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    #img[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img

class tensor_add_tensor():
    def __init__(self,a = 0.8):
        self.a1 = a
        self.a2 = 1 -self.a1

    def __call__(self, img1,img2):
        img = img1 *self.a1 + img2 * self.a2

        return img


import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


class HidePatch():
    def __init__(self, hide_prob=0.5):
        self.hide_prob = hide_prob

    def __call__(self, img):
        # 获取图像的宽度和高度
        wd, ht = img.shape[1:]

        grid_size = 8  # For cifar, the patch size is set to be 8.

        # 隐藏小块
        if grid_size > 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if random.random() <= self.hide_prob:
                        img[:,x:x_end, y:y_end] = torch.from_numpy(np.random.rand(3, x_end-x, y_end -y))

        return img


import numpy as np
from PIL import Image


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        c,h, w = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        mask = torch.tensor(mask.transpose([2,0,1]))
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        return img


class total_tensortrans():
    def __init__(self):
        self.cowmask = cowmask_tensor()
        self.re = RandomErasing(EPSILON=1,r1=0.3,sh=0.2)
        self.tat = tensor_add_tensor(a =0.7)
        self.hide_path = HidePatch(hide_prob=0.5)
    def __call__(self, alltensor):
        for batch in range(alltensor.shape[0]):
            alltensor[batch] = self.tat(alltensor[batch],alltensor[np.random.randint(low=0, high=alltensor.shape[0])])
            #alltensor[batch] = self.re(alltensor[batch])
            alltensor[batch] = self.cowmask(alltensor[batch])
            #alltensor[batch] = self.hide_path(alltensor[batch])
            #img = alltensor[batch].cpu().numpy().transpose(1,2,0)
            #cv2.imshow('test',img)
            #cv2.waitKey()
        return alltensor

class total_tensortrans_w():
    def __init__(self):
        self.cowmask = cowmask_tensor()
        self.re = RandomErasing(EPSILON=1,r1=0.3,sh=0.2)
        self.tat = tensor_add_tensor(a =0.85)
        self.hide_path = HidePatch(hide_prob=0.25)
        self.AddSaltPepperNoise = AddSaltPepperNoise(0.05)
    def __call__(self, alltensor):
        for batch in range(alltensor.shape[0]):
            a = np.random.rand()
            if a >=0.45 and a <=0.9:
              alltensor[batch] = self.tat(alltensor[batch],alltensor[np.random.randint(low=0, high=alltensor.shape[0])])
            elif a > 0.9:
              alltensor[batch] = self.AddSaltPepperNoise(alltensor[batch])
            #else:
              #alltensor[batch] = self.re(alltensor[batch])
            #alltensor[batch] = self.AddSaltPepperNoise(alltensor[batch])
            #alltensor[batch] = self.re(alltensor[batch])
            #alltensor[batch] = self.cowmask(alltensor[batch])
            #alltensor[batch] = self.hide_path(alltensor[batch])
            #img = alltensor[batch].cpu().numpy().transpose(1,2,0)
            #cv2.imshow('test',img)
            #cv2.waitKey()
        return alltensor