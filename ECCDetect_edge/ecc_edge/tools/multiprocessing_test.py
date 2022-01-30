from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
import glob
import pickle
import time

from torch import multiprocessing as mp
import threading, random, socket, socketserver, time, pickle

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from torch.autograd import Variable
from keyframe.featuremap import feature
from keyframe.keyframe_choosing import is_key, get_frames

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file', default="experiments/siamrpn_alex_dwxcorr_otb/config.yaml")
parser.add_argument('--snapshot', type=str, help='model name', default="experiments/siamrpn_alex_dwxcorr_multi/alexnet_pre.pth")
args = parser.parse_args()

def multi_processing(model, bbox, label, probs):
    """Conduct the tracker to track object according to key frame

    Args:
        model: track model
        bbox (N, 4): ground truth boxes.
        label (1, N): object labels.
        probs(1,N): Confidence

    Returns:
        res: return the track results and time consuming.
    """
    # set model on GPU
    model.eval().to('cuda')
    # build the tracker
    tracker = build_tracker(model,label,probs)
    init_rect = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
    # read images from the folder
    # image_files = sorted(glob.glob('../datasets/demo/*.JPEG'))  # the path of images
    track_start_time = time.time()
    total_time = 0
    for f, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        # initialize the tracker
        if f == 0:
            tracker.init(frame, init_rect)
            continue
        t1 = time.time()
        # track images
        output = tracker.track(frame)
        total_time += time.time()-t1
        stdOutput = [f, output, time.time()-t1]
    track_end_time = time.time()
    return [track_start_time, track_end_time, total_time, stdOutput]

def main():
    # load config
    cfg.merge_from_file(args.config)

    # create model
    model = ModelBuilder()

    # load model   
    checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        param.requires_grad = False

   # We take the following as an example of 4 processes
   bbox = [[784,173,891,291], [411,238,590,504], [196,183,421,350], [437,201,595,292]]
   label = ['zebra','lion','car','monkey']
   probs = [0.85,0.99,0.99,0.98]

    # Open process pool. It is expected that the number of processes is not greater than the number of cpu cores
    pool = mp.Pool(processes = 4)
    result = []
    
    # Execute multiprocessing asynchronously and save the results
    for i in range(len(bbox)):
        result.append(pool.apply_async(multi_processing, (model, bbox[i], label[i], probs[i])))

    # Close the process pool and wait for all processes complete
    pool.close()
    pool.join()

    # Calculate the time consuming
    total_time_log = 0
    start_time_list = []
    end_time_list = []
    log_time_list = []
    for res in result:
        time_list = res.get()
        start_time_list.append(time_list[0])
        end_time_list.append(time_list[1])
        log_time_list.append(time_list[2])
    for t in log_time_list:
        total_time_log += t

    print("Parallel time:", max(end_time_list) - min(start_time_list))
    print("Serial time:", total_time_log)


if __name__ == '__main__':
    main()

