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
import pathlib

import multiprocessing as mp
import threading, random, socket, socketserver, time, pickle

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker, build_multitracker
from torch.autograd import Variable
from keyframe.featuremap import feature
from keyframe.keyframe_choosing import is_key
from datasets.vid_dataset import ImagenetDataset
from datasets.data_preprocessing import group_annotation_by_class
#from toolkit import box_utils

from ecci_sdk import Client
from threading import Thread

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument("--dataset", type=str, default="/iccv_edge/ILSVRC2015/",
                    help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument('--config', type=str, help='config file', default="./experiments/siamrpn_alex_dwxcorr_multi/config.yaml")
parser.add_argument('--snapshot', type=str, help='model name', default="./experiments/siamrpn_alex_dwxcorr_multi/track.pth")
parser.add_argument("--label_file", type=str, default='./datasets/vid-model-labels.txt', help="The label file path.")
parser.add_argument("--eval_dir", default="./eval/output/", type=str,
                    help="The directory to store evaluation results.")

args = parser.parse_args()

def write_txt(dataset, f, bbox, label, probs):
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)

    for class_index, class_name in enumerate(class_names):
        if label == class_name:
            print(class_name)
            break
    prediction_path = eval_path / f"det_test_{class_name}.txt"
    with open(prediction_path, "a") as g:
        image_id = dataset.ids[int(f)]
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[2] < 0:
            bbox[2] = 0
        if bbox[3] < 0:
            bbox[3] = 0
        g.write(str(image_id) + " " + " " + str(probs) + " " + str(bbox[0]) + " "+ str(bbox[1])+ " "+ str(bbox[2] )+ " "+ str(bbox[3] )+ "\n")


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]

    result = areas.sum() 

     
    return result

def group_annotation_by_class(dataset):
    """ Groups annotations of dataset by class
    """
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        # print(annotation)
        # input()
        gt_boxes, classes = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i in range(0, len(classes)):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])

    return true_case_stat, all_gt_boxes


def compute_average_precision_per_class(num_true_cases, gt_boxes, prediction_file, iou_threshold, use_2007_metric):
    """ Computes average precision per class
    """
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            my_box = []
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[2]))

            my_box.append(float(t[3]))
            my_box.append(float(t[4]))
            my_box.append(float(t[5]))
            my_box.append(float(t[6]))
            box = torch.tensor(my_box).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if (image_id, max_arg) not in matched:
                    true_positive[i] = 1
                    matched.add((image_id, max_arg))
                else:
                    false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    #print(precision)
   
    return compute_average_precision(precision, recall)

def map_compute():
    dataset = ImagenetDataset(args.dataset, is_val=True)
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    true_case_stat, all_gb_boxes = group_annotation_by_class(dataset)
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            prediction_path,
            0.42,
            use_2007_metric=False
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps) / len(aps)}")
    
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

    # VID dataloader
    dataset = ImagenetDataset(args.dataset, is_val=True)
    true_case_stat, all_gb_boxes = group_annotation_by_class(dataset)

    for f in range(len(dataset)):
        frame, first_frame = dataset.get_image(f)
        if first_frame:        
            # keyframe need to be uploaded to cloud 
            print('first frame upload to cloud')

            # close the last multiprocessing
            for i in range(len(multiProcess)):
                multiProcess[i].join()

            # send frame to cloud
            payload = {"type":"data","contents":{"frame":frame}}
            print("####################",payload)
            ecci_client.publish(payload, "cloud")

            # get rect from cloud
            cloud_data = ecci_client.get_sub_data_payload_queue().get()
            print("###########recieve data from cloud",cloud_data)
            bbox= cloud_data["bbox"]
            label = cloud_data["label"]
            probs = cloud_data["probs"]

            # wirte txt
            for i in range(len(bbox)):
                write_txt(dataset, f, bbox[i], label[i], probs[i])

            # # start multiprocessing
            multiProcess = []
            for i in range(len(bbox)):
                multiProcess.append(build_multitracker(model,label[i],probs[i],resQueue))
            for i in range(len(multiProcess)):
                init_rect = [bbox[i][0],bbox[i][1],bbox[i][2]-bbox[i][0],bbox[i][3]-bbox[i][1]]
                multiProcess[i].init(frame, init_rect)
                multiProcess[i].start()
                
            key_frame = frame   
            first_frame = False
            index = 1

        # elif is_key(key_frame, frame):
        elif index % 5== 0:
            if is_key(key_frame, frame) or index % 15 ==0 :

                # keyframe need to be uploaded to cloud ##### outputs, time ######
                print('key frame upload to cloud')
            
                # close the last multiprocessing
                for i in range(len(multiProcess)):
                    multiProcess[i].join()

                # send frame to cloud
                payload = {"type":"data","contents":{"frame":frame}}
                print("####################",payload)
                ecci_client.publish(payload, "cloud")

                # get rect from cloud
                cloud_data = ecci_client.get_sub_data_payload_queue().get()
                print("###########recieve data from cloud",cloud_data)
                bbox= cloud_data["bbox"]
                label = cloud_data["label"]
                probs = cloud_data["probs"]

                
                # wirte txt
                for i in range(len(bbox)):
                    write_txt(dataset, f, bbox[i], label[i], probs[i])

                # # start multiprocessing
                multiProcess = []
                for i in range(len(bbox)):
                    multiProcess.append(build_multitracker(model,label[i],probs[i],resQueue))
                for i in range(len(multiProcess)):
                    init_rect = [bbox[i][0],bbox[i][1],bbox[i][2]-bbox[i][0],bbox[i][3]-bbox[i][1]]
                    multiProcess[i].init(frame, init_rect)
                    multiProcess[i].start()
                
                key_frame = frame
                index = 1
            else:
                print('track locally')
                for i in range(len(multiProcess)):
                    multiProcess[i].track(frame)
                    
                for i in range(len(multiProcess)):
                    resDict = resQueue.get()
                    resDict['bbox'] = [resDict['bbox'][0],resDict['bbox'][1],resDict['bbox'][0]+resDict['bbox'][2],resDict['bbox'][1]+resDict['bbox'][3]]
                    write_txt(dataset, f, resDict['bbox'], resDict['label'], resDict['probs']-0.1)
                    index += 1 

        else:
            print('track locally')
            for i in range(len(multiProcess)):
                multiProcess[i].track(frame)

            t= time.time()
            for i in range(len(multiProcess)):
                resDict = resQueue.get()
                resDict['bbox'] = [resDict['bbox'][0],resDict['bbox'][1],resDict['bbox'][0]+resDict['bbox'][2],resDict['bbox'][1]+resDict['bbox'][3]]
                write_txt(dataset, f, resDict['bbox'], resDict['label'], resDict['probs']-0.1)

            print(time.time()-t)
            index +=1
            
    map_compute()


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()

