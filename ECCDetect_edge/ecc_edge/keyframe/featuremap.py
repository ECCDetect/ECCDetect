import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from torch.autograd import Variable
import time

def feature(model,img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_size = 600
        transform = transforms.Compose([
             transforms.ToTensor()])
        image_mean = np.array([102.9801,115.9465,122.7717]) 
        print('img',img)
        img = img.astype(np.float32, copy=False)
        img -= image_mean
        img_shape = img.shape
        img_size_min = np.min(img_shape[0:2])
        img_size_max = np.max(img_shape[0:2])
        img_scale = float(image_size) / float(img_size_min)
        
        image = cv2.resize(img, None, None, fx=img_scale, fy=img_scale,
                    interpolation=cv2.INTER_LINEAR)
        image = transform(image)
        images = image.unsqueeze(0).cuda()

        return images, img_scale