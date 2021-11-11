from models.siammask_sharp import SiamMask
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr, SemanticAware
from models.mask import Mask, FCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.load_helper import load_pretrain
from resnet import resnet50
from roi_align.modules.roi_align import RoIAlignAvg
from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
import math


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        if pretrain:
            load_pretrain(self.features, 'resnet.model')

        self.downsample = ResDownS(1024, 256)

        self.layers = [self.downsample, self.features.layer2, self.features.layer3]
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]

        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x:x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return p3

    def forward_all(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return output, p3


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x, pad=0):
        return self.mask(z, x, pad)

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class MaskHEAD(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(MaskHEAD, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.RCNN_roi_align = RoIAlignAvg(self.pool_size, self.pool_size, 1.0/8.0)

    def forward(self, x, rois):
        # x = [b,256,31,31] rois = [b,100,28,28]
        # x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        pooled_feat = self.RCNN_roi_align(x, rois.view(-1, 5))
        x = self.conv1(self.padding(pooled_feat))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        #x = self.sigmoid(x)

        return x
    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

class MaskHEADRefine(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(MaskHEADRefine, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        
        self.h_conv1 = nn.Sequential( nn.Conv2d(256, 256, 3, padding=1),
                            #nn.BatchNorm2d(256, eps=0.001),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, padding=1),
                            #nn.BatchNorm2d(256, eps=0.001),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, stride=2, ceil_mode=True) )
        self.h_conv2 = nn.Sequential( nn.Conv2d(256, 256, 3, padding=1),
                            #nn.BatchNorm2d(256, eps=0.001),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, padding=1),
                            #nn.BatchNorm2d(256, eps=0.001),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, stride=2, ceil_mode=True) )
        self.v_conv1 = nn.Sequential( nn.Conv2d(256, 64, 1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 16, 1), nn.ReLU() )
        self.v_conv2 = nn.Sequential( nn.Conv2d(256, 128, 1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 64, 1), nn.ReLU() )
        self.deconv1 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        #self.RCNN_roi_align = RoIAlignAvg(self.pool_size, self.pool_size, 1.0/8.0)
        self.RCNN_roi_align = PrRoIPool2D(self.pool_size, self.pool_size, 1.0/8.0)
        

    def forward(self, x, rois):
        # x = [b,256,31,31] rois = [b,100,28,28]
        # x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        pooled_feat = self.RCNN_roi_align(x, rois.view(-1, 5)) #[b*100,256,28,28]
        
        h1 = self.h_conv1(pooled_feat)
        v1 = self.v_conv1(pooled_feat)
        
        h2 = self.h_conv2(h1)
        v2 = self.v_conv2(h1)
        
        d1 = self.deconv1(h2)
        d1 = d1 + v2

        d2 = self.deconv2(d1)
        d2 = d2 + v1
        
        output = self.conv3(d2)
        
        return output
    
    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

class Refine(nn.Module):
    def __init__(self,pool_size):
        super(Refine, self).__init__()
        self.h2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(8, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(64, 64, 1)
        self.post1 = nn.Conv2d(16, 16, 1)
        self.post2 = nn.Conv2d(4, 1, 1)

        self.RCNN_roi_align = PrRoIPool2D(pool_size, pool_size, 1.0/8.0)
        
        for modules in [self.h2, self.h1, self.h0, self.deconv, self.post0, self.post1, self.post2,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, f, corr_feature, rois, pos=None, test=False):
        pooled_feat = self.RCNN_roi_align(corr_feature, rois.view(-1, 5)) #[b*pos_num,256,14,14]

        #out = self.deconv(p3)
        out = pooled_feat
        out = self.post0(F.upsample(self.h2(out), size=(28, 28)))
        out = self.post1(F.upsample(self.h1(out), size=(56, 56)))
        out = self.post2(F.upsample(self.h0(out), size=(112, 112))) #[b*pos_num,1,112,112]
        out = out.view(-1, 112*112)
        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        #self.semantic_aware = SemanticAware(256, 256)

        #self.mask_fcn = MaskHEADRefine(256,14,255,1)
        self.refine_model = Refine(14)
    """
    def refine(self, f, pos=None):
        return self.refine_model(f, pos)
    """
    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        self.search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, self.search)
        return rpn_pred_cls, rpn_pred_loc
    """
    def track_mask(self, search):
        feature, search_feature = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(self.zf, search_feature ,padding=3)
        # search_feature = [b,256,31,31]
        # pred_cls = [b,10 h,w] pred_loc = [b,20,h,w]
        rpn_pred_cls_reshape = self.reshape(rpn_pred_cls, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_pred_cls_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, 2*5)
        rpn_rois = self.proposal_layer(rpn_cls_prob, rpn_pred_loc,True) #if testing : rpn_rois = [batch_size, 1, 5]
        mrcnn_mask_a_c = self.mask_fcn_function(corr_feature, rpn_rois)
        mrcnn_mask_b_c = self.mask_fcn_function(search_feature, rpn_rois)

        mrcnn_mask = mrcnn_mask_b_c + mrcnn_mask_a_c
        pred_mask = mrcnn_mask.view(-1,self.numP,28,28)
        
        return rpn_pred_cls, rpn_pred_loc, pred_mask, rpn_rois
    """
    def test_track_mask(self,box):
        corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search ,padding=3)
        #corr_feature = self.semantic_aware(self.zf, self.search)
        mrcnn_mask = self.mask_fcn_function(corr_feature, box)
        return mrcnn_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos, test=True)
        return pred_mask