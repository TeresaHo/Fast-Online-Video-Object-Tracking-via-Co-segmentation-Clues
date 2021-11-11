# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.anchors import generate_anchors, Anchors
from utils.tracker_config import TrackerConfig
#from utils.bbox_helper import clip_boxes

from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str

thrs = np.arange(0.3, 0.5, 0.05)

parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None, device='cpu'):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    p.renew()

    net = model
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = model.anchor_num
    p.anchor = generate_anchor(model.anchors, p.score_size)
    #single_anchor = torch.from_numpy(generate_anchors(scales=np.array([8]), 
                        #ratios=np.array([0.33, 0.5, 1, 2, 3]))).float()
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.template(z.to(device))

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state

### No voting version
"""
def siamese_track(f, state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    refine_enable = False
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        
        im_255_area = im_debug[crop_box_int[1]:crop_box_int[1] + crop_box_int[3],crop_box_int[0]:crop_box_int[0]+ crop_box_int[2],:]
        cv2.imshow('search area', im_255_area)
        cv2.waitKey(5000)
    
    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    

    if not mask_enable:   # mask_enable = True
        score, delta, mask, rois = net.track_mask(x_crop.to(device)) #rois = [batch,1,5]
    else:
        score, delta = net.track(x_crop.to(device))
    
    
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]
    # delta = [4,3125]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_sz*scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence # pscore = [3125]
    
    
    best_pscore_id = np.argmax(score)
    pred_in_crop = delta[:, best_pscore_id]
    #pred_in_crop = delta[:, best_pscore_id] / scale_x

    pred_box = np.zeros((1, 1, 5),dtype=np.uint8)
    # x1
    pred_box[:, :, 1] = pred_in_crop[0] - 0.5 * pred_in_crop[2] + 127.5
    # y1
    pred_box[:, :, 2] = pred_in_crop[1] - 0.5 * pred_in_crop[3] + 127.5
    # x2
    pred_box[:, :, 3] = pred_in_crop[0] + 0.5 * pred_in_crop[2] + 127.5
    # y2
    pred_box[:, :, 4] = pred_in_crop[1] + 0.5 * pred_in_crop[3] + 127.5

    pred_box[:, :, 1] = pred_box[:, :, 1] - 0.1*pred_in_crop[2]
    pred_box[:, :, 2] = pred_box[:, :, 2] - 0.1*pred_in_crop[3]
    pred_box[:, :, 3] = pred_box[:, :, 3] + 0.1*pred_in_crop[2]
    pred_box[:, :, 4] = pred_box[:, :, 4] + 0.1*pred_in_crop[3]

    n_w = pred_box[:, :, 3] - pred_box[:, :, 1]
    n_h = pred_box[:, :, 4] - pred_box[:, :, 2]

    def clip(boxes,im_shape):
        boxes[:,:,1].clamp_(0, im_shape-1)
        boxes[:,:,2].clamp_(0, im_shape-1)
        boxes[:,:,3].clamp_(0, im_shape-1)
        boxes[:,:,4].clamp_(0, im_shape-1)
        return boxes

    pred_box_cuda = torch.from_numpy(pred_box).type_as(x_crop).to(device)
    cliped_box = clip(pred_box_cuda, 255)
    n_w = cliped_box[:, :, 3] - cliped_box[:, :, 1]
    n_h = cliped_box[:, :, 4] - cliped_box[:, :, 2]
    

    if n_w > 0 and n_h > 0:
        mask = net.test_track_mask(cliped_box)
        mask = mask.sigmoid().squeeze().view(28, 28).cpu().data.numpy()
        mask = cv2.resize(mask, (n_w, n_h), interpolation=cv2.INTER_CUBIC)
        m_h, m_w = mask.shape
        r_mask = np.zeros((255, 255),dtype=np.float)
        cliped_box = cliped_box.cpu().data.numpy().astype(np.int64)
        r_mask[cliped_box[0,0,2]:cliped_box[0,0,2]+m_h,cliped_box[0,0,1]:cliped_box[0,0,1]+m_w] = mask
        #cv2.imshow('Sub mask', r_mask)
        #cv2.waitKey(2000)
    else:
        r_mask = np.zeros((255, 255),dtype=np.float)

    vis = False
    if  vis:
        print(im_255_area.shape)
        im_255_area = cv2.resize(im_255_area, (255, 255))
        rr_mask = (r_mask > 0.3).astype(np.uint8)
        tobend = np.stack((rr_mask, rr_mask, rr_mask), axis=2)
        tobend[:,:,0] = (tobend[:,:,0]>0.9)*250
        tobend[:,:,1] = (tobend[:,:,1]>0.9)*250
        tobend[:,:,2] = (tobend[:,:,2]>0.9)*0   
        output = cv2.addWeighted(im_255_area, 0.8, tobend.astype(np.uint8), 0.2, 0)
        
        cv2.imshow('be',output)
        cv2.waitKey(1000)
        if f<10:
            cv2.imwrite('soccer_masks_us/{:d}_mask.jpg'.format(f), output)


    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0]/ scale_x + target_pos[0]
    res_y = pred_in_crop[1]/ scale_x + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2]/ scale_x * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3]/ scale_x * lr
    def fill_mask(mask,im_sz,s_x,res_x,res_y,padding=-1):
        a = round(s_x)/255
        b = round(s_x)/255
        W = im_sz[0]
        H = im_sz[1]
        c = np.int0(crop_box[0])
        d = np.int0(crop_box[1])
        mapping = np.array([[a, 0, c],
        [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(mask, mapping, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding)
        return crop
    
    full_mask = fill_mask(r_mask,(state['im_w'], state['im_h']),s_x,res_x,res_y)
    target_mask = (full_mask > 0.2).astype(np.uint8)
    
    vis = False
    if  vis:
        tobend = np.stack((target_mask, target_mask,target_mask), axis=2)
        tobend[:,:,0] = (tobend[:,:,0]>0.9)*250
        tobend[:,:,1] = (tobend[:,:,1]>0.9)*0
        tobend[:,:,2] = (tobend[:,:,2]>0.9)*255
        output = cv2.addWeighted(im, 0.8, tobend.astype(np.uint8), 0.2, 0)
        cv2.imshow('be',output)
        cv2.waitKey(50)
        cv2.imwrite('no_v_car/{:d}_mask.jpg'.format(f), output)
    
    
    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    
    
    # for Mask Branch
    if mask_enable:
        
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = full_mask if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    return state
"""

###  Voting version (so far best)

def siamese_track(gt,gt_box, f, state, im, mask_enable=False, refine_enable=False, device='cpu', debug=False):
    refine_enable = False
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]

    if debug:
        im_debug = im.copy()
        crop_box_int = np.int0(crop_box)
        cv2.rectangle(im_debug, (crop_box_int[0], crop_box_int[1]),
                      (crop_box_int[0] + crop_box_int[2], crop_box_int[1] + crop_box_int[3]), (255, 0, 0), 2)
        
        im_255_area = im_debug[crop_box_int[1]:crop_box_int[1] + crop_box_int[3],crop_box_int[0]:crop_box_int[0]+ crop_box_int[2],:]
        cv2.imshow('search area', im_255_area)
        cv2.waitKey(5000)
    
    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    

    if not mask_enable:   # mask_enable = True
        score, delta, mask, rois = net.track_mask(x_crop.to(device)) #rois = [batch,1,5]
    else:
        score, delta = net.track(x_crop.to(device))
    
    
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]
    # delta = [4,3125]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


    # size penalty
    target_sz_in_crop = target_sz*scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence # pscore = [3125]
    
    # weighted score
    if not state['lost']:
        tmp_delta = np.zeros((4,3125),dtype=np.float64)
        tmmp_delta = np.zeros((4,3125),dtype=np.float64)
        final_score = np.zeros((3125),dtype=np.float64)
        
        tmp_delta[0,:] = (delta[0,:] - 0.5 * delta[2,:] + 127.5)/ scale_x + crop_box[0] #x1
        tmp_delta[1,:] = (delta[1,:] - 0.5 * delta[3,:] + 127.5)/ scale_x + crop_box[0] #y1
        tmp_delta[2,:] = (delta[0,:] + 0.5 * delta[2,:] + 127.5)/ scale_x + crop_box[0] #x2
        tmp_delta[3,:] = (delta[1,:] + 0.5 * delta[3,:] + 127.5)/ scale_x + crop_box[0] #y2
        last_box = state['ploygon']
        tmmp_delta[0,:] = tmp_delta[0,:] + 0.5 * delta[2,:] # cx
        tmmp_delta[1,:] = tmp_delta[1,:] + 0.5 * delta[3,:] # cy
        tmmp_delta[2,:] = tmp_delta[2,:] - 0.5 * delta[0,:] # w
        tmmp_delta[3,:] = tmp_delta[3,:] + 0.5 * delta[1,:] # h
        
        for i in range (3125):
            location = cxy_wh_2_rect(np.array([tmmp_delta[0,i], tmmp_delta[1,i]]), np.array([tmmp_delta[2,i], tmmp_delta[3,i]]))
            rbox_in_img = np.array([[location[0], location[1]],
                                        [location[0] + location[2], location[1]],
                                        [location[0] + location[2], location[1] + location[3]],
                                        [location[0], location[1] + location[3]]])
            b_overlap = vot_overlap(rbox_in_img, last_box, (im.shape[1], im.shape[0]))
            final_score[i] = 0.3*pscore[i] + 0.7*b_overlap
            #print("b_overlap")
            #print(b_overlap)
        #print("max overlap score")
        #print((final_score==pscore).all())
        #print(stop)
        #print("max score")
        #print(np.max(final_score))
        #best_pscore_id = np.argmax(final_score)
        pscore = final_score
    

    # sort score
    order = np.argsort(pscore)
    proposal2keep = order[-10:]
    
    best_pscore_id = np.argmax(pscore)
 
    
    pred_in_crop = delta[:, proposal2keep]  #[4,50]
    pred_in_crop_best = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr
    

    pred_box = np.zeros((1, 10, 5),dtype=np.uint8)
    # x1
    pred_box[:, :, 1] = pred_in_crop[0,:] - 0.5 * pred_in_crop[2,:] + 127.5
    # y1
    pred_box[:, :, 2] = pred_in_crop[1,:] - 0.5 * pred_in_crop[3,:] + 127.5
    # x2
    pred_box[:, :, 3] = pred_in_crop[0,:] + 0.5 * pred_in_crop[2,:] + 127.5
    # y2
    pred_box[:, :, 4] = pred_in_crop[1,:] + 0.5 * pred_in_crop[3,:] + 127.5

    pred_box[:, :, 1] = pred_box[:, :, 1] - 0.4*pred_in_crop[2,:]
    pred_box[:, :, 2] = pred_box[:, :, 2] - 0.4*pred_in_crop[3,:]
    pred_box[:, :, 3] = pred_box[:, :, 3] + 0.4*pred_in_crop[2,:]
    pred_box[:, :, 4] = pred_box[:, :, 4] + 0.4*pred_in_crop[3,:]

    def clip(boxes,im_shape):
        boxes[:,:,1].clamp_(0, im_shape-1)
        boxes[:,:,2].clamp_(0, im_shape-1)
        boxes[:,:,3].clamp_(0, im_shape-1)
        boxes[:,:,4].clamp_(0, im_shape-1)
        return boxes

    pred_box_cuda = torch.from_numpy(pred_box).type_as(x_crop).to(device)
    cliped_box = clip(pred_box_cuda, 255)

    
    mask = net.test_track_mask(cliped_box)
    mask = mask.sigmoid().squeeze().view(10, 28, 28).cpu().data.numpy()
    
    def fill_mask(mask,im_sz,s_x,padding=-1):
        a = round(s_x)/255
        b = round(s_x)/255
        W = im_sz[0]
        H = im_sz[1]
        c = np.int0(crop_box[0])
        d = np.int0(crop_box[1])
        mapping = np.array([[a, 0, c],
        [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(mask, mapping, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding)
        return crop
    #colors = [[255,255,0],[255,0,255],[0,255,255],[0,255,0],[255,0,0],[0,30,150]]
    cliped_box = cliped_box.cpu().data.numpy().astype(np.int64)
    all_masks = []
    for i in range(10):
        n_w = cliped_box[0, i, 3] - cliped_box[0, i, 1]
        n_h = cliped_box[0, i, 4] - cliped_box[0, i, 2]

        if n_w > 0 and n_h > 0:
            mask_candidate = cv2.resize(mask[i,:,:], (n_w, n_h), interpolation=cv2.INTER_CUBIC)
            r_mask = np.zeros((255, 255),dtype=np.float)
            
            r_mask[cliped_box[0,i,2]:cliped_box[0,i,2]+n_h,cliped_box[0,i,1]:cliped_box[0,i,1]+n_w] = mask_candidate
            #cv2.imshow('Sub mask', r_mask)
            #cv2.waitKey(2000)
        else:
            r_mask = np.zeros((255, 255),dtype=np.float)
    
        full_mask = fill_mask(r_mask,(state['im_w'], state['im_h']),s_x)
        target_mask = (full_mask > 0.2).astype(np.uint8)
        
        all_masks.append(target_mask)
        
    base_mask = np.zeros((state['im_h'], state['im_w']),dtype=np.uint8)
    for m in all_masks:
        base_mask = m + base_mask
    
    final_mask = (base_mask > 5).astype(np.uint8)
    
    res_x = pred_in_crop_best[0] + target_pos[0]
    res_y = pred_in_crop_best[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop_best[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop_best[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    vis = False
    if  vis:
        tobend = np.stack((final_mask, final_mask,final_mask), axis=2)
        tobend[:,:,0] = (tobend[:,:,0]>0.9)*255
        tobend[:,:,1] = (tobend[:,:,1]>0.9)*255
        tobend[:,:,2] = (tobend[:,:,2]>0.9)*0
        output = cv2.addWeighted(im, 0.8, tobend.astype(np.uint8), 0.2, 0)
        cx, cy, w, h = gt_box[0],gt_box[1],gt_box[2],gt_box[3]
        x0 = int(cx - w/2)
        y0 = int(cy - h/2)
        x1 = int(cx + w/2)
        y1 = int(cy + h/2)
        
        for i in range(25):
            cv2.rectangle(output, (int(cliped_box[0,i,1]/ scale_x + crop_box[0]), int(cliped_box[0,i,2]/ scale_x + crop_box[1])),
                      (int(cliped_box[0,i,3]/ scale_x + crop_box[0]), int(cliped_box[0,i,4]/ scale_x + crop_box[1])), (255, 0, 0), 1)
        cv2.rectangle(output, (int(res_x - res_w/2), int(res_y - res_h/2)),
                      (int(res_x + res_w/2), int(res_y + res_h/2)), (255, 255, 0), 2)
        cv2.rectangle(output, (x0,y0),
                      (x1,y1), (255, 0, 255), 2)
        
        cv2.imshow('final mask',output)
        cv2.waitKey(50)
        cv2.imwrite('vis/{:d}_mask.jpg'.format(f), output)
    
    target_mask = final_mask
    

    # for Mask Branch
    if mask_enable:
        
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        """ # < betetr version for VOT >
        if len(contours) != 0 and np.max(cnt_area) > 25:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle
            #print("pbox")
            #print(pbox)
            # rbox_in_img = pbox
            rbox_in_img = prbox
            #!!!!!!!!!!!!
            gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                  (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))

            location = prbox.flatten()

            pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                    (location[4], location[5]), (location[6], location[7]))
            
            b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
        else:  # empty mask
            b_overlap = 0
            
        if b_overlap:
            rbox_in_img = prbox
        else:
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        """
        
        if len(contours) != 0 and np.max(cnt_area) > 1600:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle
            # box_in_img = pbox
            rbox_in_img = prbox
            target_pos[0] = max(0, min(state['im_w'], pbox[0]))
            target_pos[1] = max(0, min(state['im_h'], pbox[1]))
            target_sz[0] = max(10, min(state['im_w'], pbox[2]))
            target_sz[1] = max(10, min(state['im_h'], pbox[3]))
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
            target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
            target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
            target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
            target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    
    """  original 
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    """
    

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score[best_pscore_id]
    state['mask'] = target_mask if mask_enable else []
    state['ploygon'] = rbox_in_img if mask_enable else []
    return state


def track_vot(model, video, hp=None, mask_enable=False, refine_enable=False, device='cpu'):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']

    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, model, hp, device)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
            pred_polygon = ((location[0], location[1]),
                                    (location[0] + location[2], location[1]),
                                    (location[0] + location[2], location[1] + location[3]),
                                    (location[0], location[1] + location[3]))
            state['ploygon'] = pred_polygon
            state['lost'] = False
        elif f > start_frame:  # tracking
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            gt_box = [cx,cy,w,h]
            state = siamese_track(gt,gt_box, f, state, im, mask_enable, refine_enable, device, args.debug)  # track
            if mask_enable:
                location = state['ploygon'].flatten()
                mask = state['mask']
            else:
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                mask = []

            if 'VOT' in args.dataset:
                gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                              (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
                if mask_enable:
                    pred_polygon = ((location[0], location[1]), (location[2], location[3]),
                                    (location[4], location[5]), (location[6], location[7]))
                else:
                    pred_polygon = ((location[0], location[1]),
                                    (location[0] + location[2], location[1]),
                                    (location[0] + location[2], location[1] + location[3]),
                                    (location[0], location[1] + location[3]))
                b_overlap = vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
            else:
                b_overlap = 1

            if b_overlap:
                state['lost'] = False
                regions.append(location)
            else:  # lost
                state['lost'] = True
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

        if args.visualization and f >= start_frame:  # visualization (skip lost frame)
            im_show = im.copy()
            if f == 0: cv2.destroyAllWindows()
            if gt.shape[0] > f:
                if len(gt[f]) == 8:
                    cv2.polylines(im_show, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                else:
                    cv2.rectangle(im_show, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
            if len(location) == 8:
                if mask_enable:
                    mask = mask > state['p'].seg_thr
                    im_show[:, :, 2] = mask * 255 + (1 - mask) * im_show[:, :, 2]
                location_int = np.int0(location)
                cv2.polylines(im_show, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            else:
                location = [int(l) for l in location]
                cv2.rectangle(im_show, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im_show, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im_show, str(state['score']) if 'score' in state else '', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(video['name'], im_show)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()

    # save result
    name = args.arch.split('.')[0] + '_' + ('mask_' if mask_enable else '') + ('refine_' if refine_enable else '') +\
           args.resume.split('/')[-1].split('.')[0]

    if 'VOT' in args.dataset:
        video_path = join('test', args.dataset, name,
                          'baseline', video['name'])
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                        fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
    else:  # OTB
        video_path = join('test', args.dataset, name)
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
        v_id, video['name'], toc, f / toc, lost_times))

    return lost_times, f / toc


def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)

    num_frame = targets.shape[0]
    if start is None:
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        object_ids = [int(id) for id in start]

    num_object = len(object_ids)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)

    output_max_id = np.argmax(outputs, axis=0).astype('uint8')+1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = targets == object_ids[j]

            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(object_ids[j])] - 1
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j+1)
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            res[j, k] = np.mean(iou)
    return res


def track_vos(model, video, hp=None, mask_enable=False, refine_enable=False, mot_enable=False, device='cpu'):
    image_files = video['image_files']

    annos = [np.array(Image.open(x)) for x in video['anno_files']]
    if 'anno_init_files' in video:
        annos_init = [np.array(Image.open(x)) for x in video['anno_init_files']]
    else:
        annos_init = [annos[0]]

    if not mot_enable:
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [(anno_init > 0).astype(np.uint8) for anno_init in annos_init]

    if 'start_frame' in video:
        object_ids = [int(id) for id in video['start_frame']]
    else:
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init*len(object_ids)
    object_num = len(object_ids)
    toc = 0
    pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0], annos[0].shape[1]))-1
    for obj_id, o_id in enumerate(object_ids):

        if 'start_frame' in video:
            start_frame = video['start_frame'][str(o_id)]
            end_frame = video['end_frame'][str(o_id)]
        else:
            start_frame, end_frame = 0, len(image_files)

        for f, image_file in enumerate(image_files):
            im = cv2.imread(image_file)
            tic = cv2.getTickCount()
            if f == start_frame:  # init
                mask = annos_init[obj_id] == o_id
                x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                cx, cy = x + w/2, y + h/2
                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])
                state = siamese_init(im, target_pos, target_sz, model, hp, device=device)  # init tracker
            elif end_frame >= f > start_frame:  # tracking
                state = siamese_track(state, im, mask_enable, refine_enable, device=device)  # track
                mask = state['mask']
            toc += cv2.getTickCount() - tic
            if end_frame >= f >= start_frame:
                pred_masks[obj_id, f, :, :] = mask
    toc /= cv2.getTickFrequency()

    if len(annos) == len(image_files):
        multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, annos,
                                            start=video['start_frame'] if 'start_frame' in video else None,
                                            end=video['end_frame'] if 'end_frame' in video else None)
        for i in range(object_num):
            for j, thr in enumerate(thrs):
                logger.info('Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(video['name'] + '_' + str(i + 1), thr,
                                                                           multi_mean_iou[i, j]))
    else:
        multi_mean_iou = []
    sav = True
    #if args.save_mask:
    if sav:
        video_path = join('test', args.dataset, 'SiamMask_rcnn', video['name'])
        if not isdir(video_path): makedirs(video_path)
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        for i in range(pred_mask_final.shape[0]):
            cv2.imwrite(join(video_path, image_files[i].split('/')[-1].split('.')[0] + '.png'), pred_mask_final[i].astype(np.uint8))

    if args.visualization:
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (np.argmax(pred_mask_final, axis=0).astype('uint8') + 1) * (
                np.max(pred_mask_final, axis=0) > state['p'].seg_thr).astype('uint8')
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask = COLORS[pred_mask_final]
        for f, image_file in enumerate(image_files):
            output = ((0.4 * cv2.imread(image_file)) + (0.6 * mask[f,:,:,:])).astype("uint8")
            cv2.imshow("mask", output)
            cv2.waitKey(1)

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f*len(object_ids) / toc))

    return multi_mean_iou, f*len(object_ids) / toc


def main():
    global args, logger, v_id
    args = parser.parse_args()
    cfg = load_config(args)

    init_log('global', logging.INFO)
    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    logger = logging.getLogger('global')
    logger.info(args)

    # setup model
    if args.arch == 'Custom':
        from custom import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        parser.error('invalid architecture: {}'.format(args.arch))

    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)
    model.eval()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    model = model.to(device)
    # setup dataset
    dataset = load_dataset(args.dataset)

    # VOS or VOT?
    if args.dataset in ['DAVIS2016', 'DAVIS2017', 'ytb_vos'] and args.mask:
        vos_enable = True  # enable Mask output
    else:
        vos_enable = False

    total_lost = 0  # VOT
    iou_lists = []  # VOS
    speed_list = []

    for v_id, video in enumerate(dataset.keys(), start=1):
        if args.video != '' and video != args.video:
            continue
        #if v_id<4:
            #continue
        if vos_enable:
            iou_list, speed = track_vos(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                                 args.mask, args.refine, args.dataset in ['DAVIS2017', 'ytb_vos'], device=device)
            iou_lists.append(iou_list)
        else:
            lost, speed = track_vot(model, dataset[video], cfg['hp'] if 'hp' in cfg.keys() else None,
                             args.mask, args.refine, device=device)
            total_lost += lost
        speed_list.append(speed)

    # report final result
    if vos_enable:
        for thr, iou in zip(thrs, np.mean(np.concatenate(iou_lists), axis=0)):
            logger.info('Segmentation Threshold {:.2f} mIoU: {:.3f}'.format(thr, iou))
    else:
        logger.info('Total Lost: {:d}'.format(total_lost))

    logger.info('Mean Speed: {:.2f} FPS'.format(np.mean(speed_list)))


if __name__ == '__main__':
    main()
