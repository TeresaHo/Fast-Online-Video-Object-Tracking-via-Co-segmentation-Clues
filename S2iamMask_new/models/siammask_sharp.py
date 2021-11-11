# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from utils.anchors import Anchors
from utils.anchors import generate_anchors, Anchors
from utils.bbox_helper import bbox_transform_inv,clip_boxes
import numpy as np
from nms.nms_wrapper import nms
import cv2
import torch.nn.functional as F


class SiamMask(nn.Module):
    def __init__(self, proposals=100, anchors=None, o_sz=127, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        #self.anchor = Anchors(anchors)
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array([8]), 
                        ratios=np.array([0.33, 0.5, 1, 2, 3]))).float()
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.numP = proposals
        #self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])

    def reshape(self, x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask_fcn_function(self, search_feature, rois):
        pred_mask = self.mask_fcn(search_feature, rois)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask, label_mask_weight)

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7
    
    def mask_loss_and_iou(self, target_mask, pre_mask):

        iou_m, iou_5, iou_7 = iou_measure(pre_mask, target_mask)
        #BCEloss = nn.BCELoss()
        #target_mask[target_mask < 0] = 0
        #loss = BCEloss(pre_mask,target_mask)
        loss = F.soft_margin_loss(pre_mask, target_mask)
        
        return loss, iou_m, iou_5, iou_7

    def run(self, template, search, gt_masks, label_loc_weight, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        feature, search_feature = self.features.forward_all(search)     # search_feature = [b,256,31,31] feature = [b,256,7,7]
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)  # pred_cls = [b,10 h,w] pred_loc = [b,20,h,w]
        #corr_feature = self.semantic_aware(template_feature, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(template_feature, search_feature ,padding=3)  # (b, 256, 31, 31)
        rpn_pred_cls_reshape = self.reshape(rpn_pred_cls, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_pred_cls_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, 2*5)
        rpn_rois = self.proposal_layer(rpn_cls_prob, rpn_pred_loc, label_loc_weight) # [b*pos_num,5]
        
        debug = False
        if  debug:
            mask = gt_masks[0].squeeze(0)
            mask = mask*255
            mask = mask.cpu().data.numpy()
            cv2.imshow('mask',mask)
            cv2.waitKey(2000)
            
            for i in range(90,100):
                top_roi = rpn_rois[0][i].cpu().data.numpy()
                x1 = int(round(top_roi[1]))
                y1 = int(round(top_roi[2]))
                x2 = int(round(top_roi[3]))
                y2 = int(round(top_roi[4]))
                mask_crop = mask[y1:y2,x1:x2]
                if mask_crop.shape:
                    cv2.imshow('mask_crop',mask_crop)
                    cv2.waitKey(3000)
                else:
                    print("empty!!!")
        if self.training:
            target_mask = self.detection_target_layer(rpn_rois, gt_masks, label_loc_weight) #[b*pos_num,112,112]
            
        #mrcnn_mask = self.mask_fcn_function(corr_feature, rpn_rois)
        mrcnn_mask = self.refine(feature, corr_feature,rpn_rois) #[-1,112,112]
        #mrcnn_mask = mrcnn_mask.view(-1,self.numP,127,127)
        
        return rpn_pred_cls, rpn_pred_loc, mrcnn_mask, target_mask
        """
        corr_feature = self.mask_model.mask.forward_corr(template_feature, search_feature)  # (b, 256, w, h)
        rpn_pred_mask = self.refine_model(feature, corr_feature)

        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature
        """

    def detection_target_layer(self,rpn_rois, gt_masks, loc_weight):
        """
        crop gt_mask based on given position and resize it to 28,28
        """
        
        rois = rpn_rois.cpu().data.numpy()
        batch_size = rpn_rois.size(0)
        target_masks = rpn_rois.new(batch_size, self.numP, 127,127).zero_()
        mask_uf = F.unfold(gt_masks, (112, 112), padding=1, stride=6) # [b, 112*112, 25*25]
        mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, 25*25, 112 * 112) #[b, 25*25, 112*112]
        
        mask_uf = mask_uf.repeat(1,5,1) #[b, 5*25*25, 112*112]
        mask_uf = mask_uf.view(-1,112*112)
        loc_weight = (loc_weight > 0) #[b,5,25,25]
        loc_weight = loc_weight.view(-1)
        pos = Variable(loc_weight.data.eq(1).nonzero().squeeze())
        target_masks = torch.index_select(mask_uf, 0, pos)
        
        #overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        #max_overlaps, gt_assignment = torch.max(overlaps, 2)
        """
        for i in range(batch_size):
            weight = loc_weight[i].view(-1)
            pos = Variable(weight.data.eq(1).nonzero().squeeze())
            if pos.size(0) > post_nms_topN:
                pos = pos[:post_nms_topN]
            m = mask_uf[i]
            selected_mask = torch.index_select(m, 0, pos) #[pos_num,112*112]
            target_masks[i,:pos.size(0),:] = selected_mask
        """
        """
        for i in range(batch_size):

            #fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            #fg_num_rois = fg_inds.numel()

            roi_num = rpn_rois[i].size(0)
            for j in range(roi_num):
                tmp = torch.zero(127,127)
                x1 = int(round(rois[i,j,1]))
                y1 = int(round(rois[i,j,2]))
                x2 = int(round(rois[i,j,3]))
                y2 = int(round(rois[i,j,4]))
                cx = int(( x1 + x2 ) / 2)
                cy = int(( y1 + y2 ) / 2)
                
                l = max((cx - 63),0)
                r = min((cx + 63), 254)
                u = max((cy - 63),0)
                b = min((cy + 63),254)
                target = gt_masks[i,0,u:b,l:r]
                
                if not target.nelement()==0:
                    target = target.unsqueeze(0).unsqueeze(0)
                    rz_target = F.interpolate(target, size=28, mode ='bilinear')
                    output[i,j,:,:] = rz_target
        """
        return target_masks

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            label_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']
            gt_bbox = input['gt_bbox']
        # gt box = [b,4]
        # [b, 5, 25, 25]
        
        
        
        rpn_pred_cls, rpn_pred_loc, mrcnn_mask, target_mask = self.run(template, search, label_mask, label_loc_weight, softmax=self.training)
        # target_mask = [b, 5*25*25, 112*112]
        outputs = dict()

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, mrcnn_mask]

        if self.training:
            #rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                #self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   #rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
            loss, iou_m, iou_5, iou_7 = self.mask_loss_and_iou(target_mask, mrcnn_mask)
            outputs['losses'] = [loss]
            outputs['accuracy'] = [iou_m, iou_5, iou_7]

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc

    def proposal_layer(self, rpn_pred_cls, rpn_pred_loc, loc_weight, testing=False):
        """Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinment detals to anchors.

        Inputs:
            rpn_probs: [batch, anchors, (bg prob, fg prob)]
            rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
        """
        batch_size = rpn_pred_cls.size(0)

        feat_height, feat_width = 25,25
        shift_x = np.arange(0, feat_width) * 8
        shift_y = np.arange(0, feat_height) * 8
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_pred_cls).float()

        A = 5
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(rpn_pred_cls)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        
        pre_nms_topN = 1000
        post_nms_topN = self.numP
        nms_thresh = 0.7

        
        scores = rpn_pred_cls[:,5:,:,:] #[b,5,25,25]
        deltas = rpn_pred_loc  #[b,20,25,25]

        bbox_deltas = deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)
        ### try first using anchor without refinement
        #proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        proposals = scores.new(batch_size, 5*25*25, 5).zero_()
        ### don't apply delta when training !!!
        #proposals = clip_boxes(proposals, 255)   #proposals = [b,5*25*25,5]
        proposals[:,:,1:5] = anchors
        proposals = proposals.view(-1,5)
        loc_weight = (loc_weight > 0) #[b,5,25,25]
        loc_weight = loc_weight.view(-1)
        pos = Variable(loc_weight.data.eq(1).nonzero().squeeze())
        selected_proposals = torch.index_select(proposals, 0, pos)

        return selected_proposals
        
        """
        scores_keep = scores
        
        _, order = torch.sort(scores_keep, 1, True)
        if testing:
            output = scores.new(batch_size, 1, 5).zero_()
        else:
            output = scores.new(batch_size, post_nms_topN, 5).zero_()
        """
        
        """
        for i in range(batch_size):
            weight = loc_weight[i].view(-1)
            pos = Variable(weight.data.eq(1).nonzero().squeeze())
            if pos.size(0) > post_nms_topN:
                pos = pos[:post_nms_topN]
            p = proposals[i]
            selected_proposals = torch.index_select(p, 0, pos) #[pos_num,4]
            output[i,:,0] = i
            output[i,:pos.size(0),1:] = selected_proposals
        """

        """
        proposals_keep = proposals
          
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            if testing:
                best_id = order[i,0]
                best_proposal = proposals_keep[i,best_id,:]
                x1 = best_proposal[0]
                y1 = best_proposal[1]
                x2 = best_proposal[2]
                y2 = best_proposal[3]
                h = y2 - y1
                w = x2 - x1
                x1 = x1 - 0.3*w
                x2 = x2 + 0.3*w
                y1 = y1 - 0.3*h
                y2 = y2 + 0.3*h
                enlarged = torch.tensor([x1,y1,x2,y2])
                enlarged = enlarged.view(1,1,4)
                cliped = clip_boxes(enlarged, 255)
                output[i,:,0] = i
                output[i,:,1:] = cliped
                return output

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]
            
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            
            
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single
        """
        
        # Normalize dimensions to range of 0 to 1.
        """
        norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
        if config.GPU_COUNT:
            norm = norm.cuda()
        normalized_boxes = output / norm
        """

        #return output

def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps

def get_cls_loss(pred, label, select):
    if select.nelement() == 0: return pred.sum()*0.
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)

    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        p_m = p_m.view(-1, g_sz * g_sz)
    else:
        p_m = torch.index_select(p_m, 0, pos)

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=0, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

    mask_uf = torch.index_select(mask_uf, 0, pos)
    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    pred = pred.view(-1,28*28)
    label = label.view(-1,28*28)
    pred = pred.ge(0)
    mask_sum = pred.eq(1).add(label.eq(1))
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn/union
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])
    

if __name__ == "__main__":
    p_m = torch.randn(4, 63*63, 25, 25)
    cls = torch.randn(4, 1, 25, 25) > 0.9
    mask = torch.randn(4, 1, 255, 255) * 2 - 1

    loss = select_mask_logistic_loss(p_m, mask, cls)
    print(loss)
