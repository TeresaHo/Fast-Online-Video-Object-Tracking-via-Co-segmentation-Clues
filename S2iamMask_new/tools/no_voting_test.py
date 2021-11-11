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
        tobend[:,:,2] = (tobend[:,:,2]>0.9)*250  
        output = cv2.addWeighted(im, 0.8, tobend.astype(np.uint8), 0.2, 0)
        cv2.imshow('be',output)
        cv2.waitKey(50)
        #cv2.imwrite('exe_masks/{:d}_mask.jpg'.format(f), output)
    
    
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



### function with voting system
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
        cv2.imshow('search area', im_debug)
        cv2.waitKey(6000)

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    if not mask_enable:
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
    
    # sort score
    order = np.argsort(pscore)
    proposal2keep = order[-50:]
    
    best_pscore_id = np.argmax(pscore)
    pred_in_crop = delta[:, proposal2keep]  #[4,50]
    #pred_in_crop = delta[:, best_pscore_id] / scale_x
    
    pred_box = np.zeros((1, 50, 5),dtype=np.uint8)
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

    n_w = pred_box[:, :, 3] - pred_box[:, :, 1]
    n_h = pred_box[:, :, 4] - pred_box[:, :, 2]
    
    def clip(boxes,im_shape):
        boxes[:,:,1].clamp_(0, im_shape-1)
        boxes[:,:,2].clamp_(0, im_shape-1)
        boxes[:,:,3].clamp_(0, im_shape-1)
        boxes[:,:,4].clamp_(0, im_shape-1)
        return boxes

    pred_box_cuda = torch.from_numpy(pred_box).type_as(x_crop).to(device)
    cliped_box = clip(pred_box_cuda, 255)  # shape=[1,50,5]

    mask = net.test_track_mask(cliped_box)
    mask = mask.sigmoid().squeeze().view(50, 28, 28).cpu().data.numpy()

    def fill_mask(mask,im_sz,s_x,res_x,res_y,padding=-1):
        a = s_x/255
        b = s_x/255
        W = im_sz[0]
        H = im_sz[1]
        c = np.int0(res_x - s_x/2)
        d = np.int0(res_y - s_x/2)
        mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(mask, mapping, (W, H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=padding)
        return crop
    cliped_box = cliped_box.cpu().data.numpy().astype(np.int64)
    all_masks = []
    for i in range(50):
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

        lr = penalty[proposal2keep] * score[proposal2keep] * p.lr  # lr for OTB

        res_x = pred_in_crop[0,i] + target_pos[0]
        res_y = pred_in_crop[1,i] + target_pos[1]
        
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2,i] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3,i] * lr
    
        full_mask = fill_mask(r_mask,(state['im_w'], state['im_h']),s_x,res_x,res_y)
        target_mask = (full_mask > 0.3).astype(np.uint8)
        #target_mask = (full_mask > 0.3).astype(np.uint8)*255
        all_masks.append(target_mask)
        #cv2.imshow('Full mask in img', target_mask)
        #cv2.waitKey(1000)
    base_mask = np.zeros((state['im_h'], state['im_w']),dtype=np.uint8)
    for m in all_masks:
        base_mask = m + base_mask
    
    final_mask = (base_mask > 40).astype(np.uint8)*255
    #cv2.imshow('Full mask in img', final_mask)
    #cv2.waitKey(1000)
    
    vis = True
    if  vis:
        tobend = np.stack((target_mask, target_mask,target_mask), axis=2)
        tobend[:,:,0] = (tobend[:,:,0]>0.9)*0
        tobend[:,:,1] = (tobend[:,:,1]>0.9)*0
        tobend[:,:,2] = (tobend[:,:,2]>0.9)*255   
        output = cv2.addWeighted(im, 0.8, tobend.astype(np.uint8), 0.2, 0)
        cv2.imshow('be',output)
        cv2.waitKey(1000)
        #cv2.imwrite('scooter_mask/{:d}_mask.jpg'.format(f), output)
    
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr
    res_w_final = target_sz[0] * (1 - lr) + delta[:, best_pscore_id][2] * lr
    res_h_final = target_sz[1] * (1 - lr) + delta[:, best_pscore_id][3] * lr
    res_x_final = delta[:, best_pscore_id][0] + target_pos[0]
    res_y_final = delta[:, best_pscore_id][1] + target_pos[1]
    
    target_pos = np.array([res_x_final, res_y_final])
    target_sz = np.array([res_w_final, res_w_final])
    
    # for Mask Branch
    if mask_enable:
        """
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if refine_enable:
            mask = net.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop


        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))
        
        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        """
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