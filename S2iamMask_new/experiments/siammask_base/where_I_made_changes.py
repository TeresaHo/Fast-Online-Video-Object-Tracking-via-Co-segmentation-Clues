# 
"""
cumtom.py line 89  
MaskCorr
add generate attention function
add line 95~98
line101  #return self.mask(z,x)
"""

"""
in siam_mask_dataset
line 581 search, bbox, mask = self.search_aug(search_image, search_box, self.search_size, gray=gray, mask=search_mask)
=> search, bbox, _ = self.search_aug(search_image, search_box, self.search_size, gray=gray, mask=search_mask)
line 582
add exchange z and x
line 610 rerturn no mask and add return template2 search2

in train_siammask
line 226 
X= {
    delete mask
    add tem2 serch2...
}
orifinal x = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[0]).cuda(),
            'search': torch.autograd.Variable(input[1]).cuda(),
            'label_cls': torch.autograd.Variable(input[2]).cuda(), #[64,5,25,25]
            'label_loc': torch.autograd.Variable(input[3]).cuda(), ##[64,4,5,25,25]
            'label_loc_weight': torch.autograd.Variable(input[4]).cuda(), #[64,5,25,25]
            'label_mask': torch.autograd.Variable(input[6]).cuda(),
            'label_mask_weight': torch.autograd.Variable(input[7]).cuda(),
        }
in sisammask.py
add def co_attention(mask1,mask2,img1,img2)
add def select_mask()

in Class custom
line 129 add self.post_extractor


line 135
in self._add_rpn_loss  no pass mask_label

in siammask.py
line 133
if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)  ==> no label_mask passed

self.add_rpn_loss
"""