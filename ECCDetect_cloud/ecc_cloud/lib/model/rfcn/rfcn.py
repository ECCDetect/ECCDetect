import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from model.psroi_pooling.modules.psroi_pool import PSRoIPool
from model.roi_layers import PSROIPool as PSRoIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import _smooth_l1_loss

class _RFCN(nn.Module):
    """ R-FCN """
    def __init__(self, classes, class_agnostic):
        super(_RFCN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RFCN_loss_cls = 0
        self.RFCN_loss_bbox = 0

        self.box_num_classes = (1 if class_agnostic else len(classes))
        #print(self.box_num_classes)
        #input()
        # define rpn
        self.RFCN_rpn = _RPN(self.dout_base_model)
        self.RFCN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RFCN_psroi_pool_cls = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1./16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.n_classes)
        self.RFCN_psroi_pool_loc = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1./16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.box_num_classes * 4)
        #self.pooling = nn.AvgPool2d(kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RFCN_cls_net = nn.Conv2d(512,self.n_classes*7*7, [1,1], padding=0, stride=1)
        nn.init.normal(self.RFCN_cls_net.weight.data, 0.0, 0.01)
        
        self.RFCN_bbox_net = nn.Conv2d(512, 4*self.box_num_classes*7*7, [1,1], padding=0, stride=1)
        nn.init.normal(self.RFCN_bbox_net.weight.data, 0.0, 0.01)

        self.RFCN_cls_score = nn.AvgPool2d((7,7), stride=(7,7))
        #print(self.RFCN_cls_score)
        #input()
        self.RFCN_bbox_pred = nn.AvgPool2d((7,7), stride=(7,7))


    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        #print(batch_size)
        #input()

        im_info = im_info
        gt_boxes = gt_boxes
        #print("gt_boxes", gt_boxes)
        #input()
        num_boxes = num_boxes
        #self.batch_size = im_data.size(0)

        # feed image data to base model to obtain base feature map
        #base_feat = self.RFCN_base(im_data)
        base_feat = self._im_to_head(im_data)
        rfcn_cls = self.RFCN_cls_net(base_feat)
        rfcn_bbox = self.RFCN_bbox_net(base_feat)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RFCN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RFCN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        #base_feat = self.RFCN_conv_new(base_feat)

        # do roi pooling based on predicted rois
        #cls_feat = self.RFCN_cls_base(base_feat)
        pooled_feat_cls = self.RFCN_psroi_pool_cls(rfcn_cls, rois.view(-1, 5))
        pooled_feat_loc = self.RFCN_psroi_pool_loc(rfcn_bbox, rois.view(-1, 5))
        cls_score = self.RFCN_cls_score(pooled_feat_cls).squeeze()
        cls_prob = F.softmax(cls_score, dim=1)

        #bbox_base = self.RFCN_bbox_base(base_feat)
        #pooled_feat_loc = self.pooling(pooled_feat_loc)
        bbox_pred =self.RFCN_bbox_pred(pooled_feat_loc).squeeze()

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        RFCN_loss_cls = torch.zeros(1).cuda()
        RFCN_loss_bbox = torch.zeros(1).cuda()

        #cls_prob = F.softmax(cls_score, dim=1)

        #RFCN_loss_cls = 0
        #RFCN_loss_bbox = 0

        if self.training:
            RFCN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RFCN_loss_bbox = _smooth_l1_loss(bbox_pred, 
                rois_target, rois_inside_ws, rois_outside_ws)
            #loss_func = self.ohem_detect_loss if cfg.TRAIN.OHEM else self.detect_loss
            #RFCN_loss_cls, RFCN_loss_bbox = loss_func(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RFCN_loss_cls, RFCN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.RFCN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RFCN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RFCN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RFCN_conv_1x1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RFCN_cls_base, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RFCN_bbox_base, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
