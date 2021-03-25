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

import multiprocessing as mp
import threading, random, socket, socketserver, time, pickle

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker, build_multitracker
from keyframe.keyframe_choosing import is_key
from ecci_sdk import Client
from threading import Thread
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file', default="./experiments/siamrpn_alex_dwxcorr_multi/config.yaml")
parser.add_argument('--snapshot', type=str, help='model name', default="./experiments/siamrpn_alex_dwxcorr_multi/alexnet_pre.pth")
args = parser.parse_args()


    
def main():
    # Initialize ecci sdk and connect to the broker in edge-cloud
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    print('edge start --------')

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model   
    checkpoint = torch.load(args.snapshot)
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)

    #multiprocessing
    manager = mp.Manager()
    resQueue = manager.Queue()
    multiProcess = []
    label = []
    probs = []

    for i in range(10):
        multiProcess.append(build_multitracker(model,label,probs,resQueue))
        multiProcess[i].start()

    first_frame = True
    image_files = sorted(glob.glob('./test/image/*.JPEG'))

    for f, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        
        if first_frame:        
            # keyframe need to be uploaded to cloud 
            print('first frame')

            payload = {"type":"data","contents":{"frame":frame}}
            print("####################",payload)
            ecci_client.publish(payload, "cloud")

            cloud_data = ecci_client.get_sub_data_payload_queue().get()
            print("###########recieve data from cloud",cloud_data)
            bbox= cloud_data["bbox"]
            label = cloud_data["label"]
            probs = cloud_data["probs"]
            num_process = len(bbox)

            t_detect_start = time.time()
            for i in range(num_process):
                cv2.rectangle(frame, (int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),(255, 255, 255), 3)
                init_rect = [bbox[i][0],bbox[i][1],bbox[i][2]-bbox[i][0],bbox[i][3]-bbox[i][1]]
                multiProcess[i].init(frame, init_rect, label[i], probs[i])
            t_detect_end = time.time()
            print("detect fps : ", 1/(t_detect_end - t_detect_start))
                           
            key_frame = frame   
            first_frame = False
            index = 1

        elif index % 10 == 0:
            if is_key(key_frame, frame) or index % 20 ==0 :
                # keyframe need to be uploaded to cloud ##### outputs, time ######
                print('key frame')
            
                payload = {"type":"data","contents":{"frame":frame}}
                print("####################",payload)
                ecci_client.publish(payload, "cloud")

                cloud_data = ecci_client.get_sub_data_payload_queue().get()
                print("###########recieve data from cloud",cloud_data)
                bbox= cloud_data["bbox"]
                label = cloud_data["label"]
                probs = cloud_data["probs"]
                num_process = len(bbox)
                
                t_detect_start = time.time()
                for i in range(num_process):
                    cv2.rectangle(frame, (int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),(255, 255, 0), 3)
                    init_rect = [bbox[i][0],bbox[i][1],bbox[i][2]-bbox[i][0],bbox[i][3]-bbox[i][1]]
                    multiProcess[i].init(frame, init_rect, label[i], probs[i])
                t_detect_end = time.time()
                print("detect fps : ", 1/(t_detect_end - t_detect_start))
                    
                key_frame = frame
                index = 1
            else:
                print('non-key frame')
                t_track_start = time.time()
                for i in range(num_process):
                    multiProcess[i].track(frame)
                t_track_end = time.time()
                print("track fps : ", 1/(t_track_end - t_track_start))

                for i in range(num_process):
                    resDict = resQueue.get()
                    print(resDict)
                    bbox = list(map(int, resDict['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3]),(0, 255, 0), 3)
                index += 1

        else:
            print('non-key frame')
            t_track_start = time.time()
            for i in range(num_process):
                multiProcess[i].track(frame)
            t_track_end = time.time()
            print("track fps : ", 1/(t_track_end - t_track_start))

            for i in range(num_process):
                resDict = resQueue.get()
                print(resDict)
                bbox = list(map(int, resDict['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3]),(0, 255, 0), 3)
            index += 1

        cv2.imwrite('./test/output/%s.jpg'%f, frame)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()

