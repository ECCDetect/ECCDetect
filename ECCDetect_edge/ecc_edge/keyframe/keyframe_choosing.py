import os
import argparse

import cv2
import torch
import numpy as np
import glob
import pickle


def dHash(img):
    img = cv2.resize(img, (16, 15), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    return dhash_str

def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def is_key(img1, img2):

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    
    n = cmpHash(hash1, hash2)
    result = 1-n/225
    if result < 0.8:
        print(result)
        key_frame = img2
        return True
    else:
        return False

def variance_of_laplacian(image):
	'''
    计算图像的laplacian响应的方差值
    '''
	return cv2.Laplacian(image, cv2.CV_64F).var()