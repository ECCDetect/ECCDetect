#!/usr/bin/python3
"""Script for creating classes for data preprocessing. It has three classes corresponding to training, testing and prediction.
"""
import torch 

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