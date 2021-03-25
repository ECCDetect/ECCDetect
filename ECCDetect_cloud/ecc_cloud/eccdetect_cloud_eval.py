# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
import pdb

from ecci_sdk import Client
import threading
from threading import Thread

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--arch', dest='arch', default='rfcn', choices=['rcnn', 'rfcn', 'couplenet'])
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='imagenet_vid', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=100, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=1036, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--load_name', 
                        default="./models/detect.pth",
                        help='load checkpoint',
                        )
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def bbox_detections(class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    result = []
    probs = []
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            result.append(bbox)
            probs.append(score)

    return result, probs

def feature(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_size = 600
        transform = transforms.Compose([
             transforms.ToTensor()])
        image_mean = np.array([102.9801,115.9465,122.7717]) 
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

if __name__ == '__main__':
    
    # Initialize ecci sdk and connect to the broker in edge-cloud
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    print('cloud start --------')

    args = parse_args()

    if args.arch == 'rcnn':
        from model.faster_rcnn.vgg16 import vgg16
        from model.faster_rcnn.resnet import resnet
    elif args.arch == 'rfcn':
        from model.rfcn.resnet_atrous import resnet
    elif args.arch == 'couplenet':
        from model.couplenet.resnet_atrous import resnet

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet_vid":
        args.imdb_name = "imagenet_vid_train"
        args.imdbval_name = "imagenet_vid_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet_vid+imagenet_det":
        args.imdb_name = "imagenet_vid_train+imagenet_det_train"
        args.imdbval_name = "imagenet_vid_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb = ['__background__', 'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 
    'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 
    'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 
    'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']

    load_name = args.load_name
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")

    fasterRCNN.create_architecture()  ######create model
    print("load checkpoint %s" % (load_name))
    
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'],strict=False)
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    if args.cuda:
        cfg.CUDA = True
    if args.cuda:
        fasterRCNN.cuda()
    max_per_image = 100
    vis = args.vis
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

   
    fasterRCNN.eval()   

############################################### model  completed#############################################

###########################################    data detection         #######################################
    while True:
        track_bbox_result = []   #detection result for track
        track_label_result = []
        track_probs_result = []
        
        edge_data = ecci_client.get_sub_data_payload_queue().get()
        
        frame = edge_data["frame"]
        images, img_scale = feature(frame)
        im_data = images
        im_info = np.multiply(frame.shape[0:2],img_scale).tolist()
        im_info.append(img_scale)
        im_info = torch.Tensor([im_info]).cuda()

        gt_boxes = torch.Tensor([[1.,1.,1.,1.,1.]]).cuda() ######constant, 1##############
        num_boxes = torch.Tensor([0]).cuda()                ######constant, 0#############
        

        det_tic = time.time()
        start = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)##### preditct result######

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]


        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(args.batch_size, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(imdb))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))


    
        pred_boxes /= (im_info[0][2])   #### box result in original image###

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic  #####detect time##########

        misc_tic = time.time()


        for j in xrange(1, len(imdb)):  ######## nms ########
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
    
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
            
                result, probs = bbox_detections(imdb[j], cls_dets.cpu().numpy(), 0.1)###############track need result#######
                if len(result) != 0:
                    for m in range(len(result)): ######  updata result for each detection###
                        track_bbox_result.append(result[m])
                        track_label_result.append(imdb[j])
                        track_probs_result.append(probs[m])

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic  ##### nms time ######

        print('im_detect:  {:.3f}s {:.3f}s   \r' \
                            .format( detect_time, nms_time))
                            
        payload = {"type":"data","contents":{"bbox":track_bbox_result,"label":track_label_result,"probs":track_probs_result}}
        print("###########send boxes to edge",payload)
        ecci_client.publish(payload, "edge")

############################################               data detection completed                    #####################################################