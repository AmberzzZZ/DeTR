import tensorflow as tf
import keras.backend as K
from scipy.optimize import linear_sum_assignment


def detr_loss(args, n_classes, cost_cls=1, cost_bbox=5, cost_giou=2):

    cls_preds, box_preds, targets = args

    # on fg targets, choose the best match in preds, take the rest as non-obj
    # best match: min(overall cost)
    # cost_cls/bbox/cost_giou: cost weights

    # preds:  [b,N1,cls+4]
    # target: [b,N2,cls+4]

    # tranverse per sample
    bs = tf.shape(cls_preds)[0]    # batch size
    n_fg = tf.TensorArray(tf.float32, size=1, dynamic_size=True)  # 动态size数组
    cls_loss = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    l1_loss = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    giou_loss = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    def loop_body(b, n_fg, cls_loss, l1_loss, giou_loss):
        pred_prob = cls_preds[b]
        pred_bbox = box_preds[b]    # [N1,4]

        target_cls = targets[b][:, :n_classes]
        target_bbox = targets[b][:, n_classes:]
        fg_indices = tf.where(target_cls[:,-1]<1)   # fg indices among N-dim
        target_cls = tf.gather_nd(target_cls, fg_indices)     # [N2,cls]
        target_bbox = tf.gather_nd(target_bbox, fg_indices)   # [N2,4]

        # cls_cost: -out_prob, (N1,N2)
        target_cls_id = tf.argmax(target_cls, axis=1)   # [N2,]
        cls_cost = -tf.gather(pred_prob, target_cls_id, axis=1)   # [N1,N2]

        # bbox_cost: torch.cdist, similarity matrix, (N1,N2)
        bbox_cost = cdist(pred_bbox, target_bbox, mode='L1')

        # giou_cost: giou
        giou_cost = -giou(pred_bbox, target_bbox)   # [N1,N2]

        # total cost
        total_cost = cost_cls * cls_cost + cost_bbox * bbox_cost + cost_giou * giou_cost   # [N1,N2]

        # hungarian
        # total_cost_mem = K.get_value(total_cost)   # into arr
        # indices = linear_sum_assignment(total_cost_mem)
        id1, id2 = tf_linear_sum_assignment(total_cost)    # arr with shape (N2,)
        id1 = tf.reshape(id1, (-1,1))   # row indices, (N2,1)
        id2 = tf.reshape(id2, (-1,1))   # row indices

        # take matches to compute loss
        matched_pred_prob = tf.gather_nd(cls_preds[b], id1)   # [N2,cls]
        matched_pred_bbox = tf.gather_nd(box_preds[b], id1)   # [N2,4]
        matched_targets = tf.gather_nd(targets[b], id2)    # [N2,cls+4]
        # # fordebug: pad to 10 for standardize output
        # gap = 5 - tf.shape(matched_targets)[0]
        # matched_targets = tf.pad(matched_targets, [[0,gap],[0,0]])

        # cls_loss: ce
        sample_cls_loss = loss_cls(matched_targets[:,:n_classes], matched_pred_prob)

        # bbox_loss: l1 + giou
        sample_l1_loss = loss_l1(matched_targets[:,-4:], matched_pred_bbox)
        id12 = tf.concat([id1,id2],axis=1)   # [N2,2]
        matched_giou = tf.gather_nd(giou_cost, id12)   # [N2,], -giou, [-1,1]
        sample_giou_loss = K.sum(1 + matched_giou)

        # cardinality_loss: count the number of bgs in original code
        # pass

        # add to list
        n_fg = n_fg.write(b, tf.cast(tf.shape(target_cls)[0], dtype=tf.float32))
        cls_loss = cls_loss.write(b, sample_cls_loss)
        l1_loss = l1_loss.write(b, sample_l1_loss)
        giou_loss = giou_loss.write(b, sample_giou_loss)
        return b+1, n_fg,cls_loss,l1_loss,giou_loss

    # run while loop
    _, n_fg,cls_loss,l1_loss,giou_loss = K.control_flow_ops.while_loop(lambda b,*args: b<bs, loop_body, [0, n_fg,cls_loss,l1_loss,giou_loss])

    # merge loss
    total_n_fg = K.sum(n_fg.stack())   # [b,] each ele=sum_fg
    total_cls_loss = cls_loss.stack() / total_n_fg
    total_l1_loss = l1_loss.stack() / total_n_fg
    total_giou_loss = giou_loss.stack() / total_n_fg

    total_loss = total_cls_loss + total_l1_loss + total_giou_loss

    # return [total_cls_loss, total_l1_loss, total_giou_loss, total_loss]
    return total_loss


def detr_loss_output(inputs):
    return [(None,)] * 4


def matching(matrix_batch):

    def hungarian(x):
        # x: [N1,N2]
        indices = linear_sum_assignment(x)  # [N2,N2]
        return indices.astype(np.int32)

    listperms = tf.py_func(hungarian, [matrix_batch], [tf.int32, tf.int32])
    return listperms


def tf_linear_sum_assignment(cost_mat):
    return tf.py_func(linear_sum_assignment, [cost_mat], [tf.int64, tf.int64])


def loss_cls(y_true, y_pred):
    # total ce on matched fg boxes per sample
    # y_true: [N,cls]
    # y_pred: [N,cls]
    pt = K.abs(y_true - y_pred)
    pt = K.clip(pt, K.epsilon(), 1-K.epsilon())
    ce = -K.log(1-pt)
    return K.sum(K.mean(ce, axis=-1))   # scalar


def loss_l1(y_true, y_pred):
    # total l1 on matched fg boxes per sample
    # y_true: [N,4]
    # y_pred: [N,4]
    l1 = K.abs(y_true-y_pred)
    return K.sum(K.mean(l1, axis=-1))   # scalar


def cdist(tensor1, tensor2, mode='L1'):
    # expand dim
    tensor1 = tf.expand_dims(tensor1, axis=1)
    tensor2 = tf.expand_dims(tensor2, axis=0)
    if mode=='L1':
        dist = tf.abs(tensor1 - tensor2)  # [N1,N2,D]
        sim = K.sum(dist, axis=2)    # [N1,N2]
    elif mode=='L2':
        dist = tf.square(tf.abs(tensor1 - tensor2))  # [N1,N2,D]
        sim = tf.sqrt(K.sum(dist, axis=2))   # [N1,N2]
    return sim


def giou(pred_boxes, gt_boxes):

    # xcycwh -> x1y1x2y2
    pred_x1y1 = pred_boxes[:,:2] - pred_boxes[:,2:]/2.
    pred_x2y2 = pred_boxes[:,:2] + pred_boxes[:,2:]/2.
    pred_x1, pred_y1 = tf.split(tf.expand_dims(pred_x1y1, axis=1), 2, axis=-1)   # [N1,1,1]
    pred_x2, pred_y2 = tf.split(tf.expand_dims(pred_x2y2, axis=1), 2, axis=-1)

    gt_x1y1 = gt_boxes[:,:2] - gt_boxes[:,2:]/2.
    gt_x2y2 = gt_boxes[:,:2] + gt_boxes[:,2:]/2.
    gt_x1, gt_y1 = tf.split(tf.expand_dims(gt_x1y1, axis=0), 2, axis=-1)   # [1,N2,1]
    gt_x2, gt_y2 = tf.split(tf.expand_dims(gt_x2y2, axis=0), 2, axis=-1)

    # iou
    inter_w = K.maximum(0., K.minimum(gt_x2, pred_x2) - K.maximum(gt_x1, pred_x1))
    inter_h = K.maximum(0., K.minimum(gt_y2, pred_y2) - K.maximum(gt_y1, pred_y1))
    inter = inter_w * inter_h

    area_gt = K.maximum(0., (gt_x2-gt_x1)*(gt_y2-gt_y1))
    area_pred = K.maximum(0., (pred_x2-pred_x1)*(pred_y2-pred_y1))
    union = area_gt + area_pred - inter + 1e-16

    iou = inter / union

    # giou
    convex_w = K.maximum(gt_x2, pred_x2) - K.minimum(gt_x1, pred_x1)
    convex_h = K.maximum(gt_y2, pred_y2) - K.minimum(gt_y1, pred_y1)
    c_area = convex_w * convex_h + 1e-16

    giou = iou - (c_area - union) / c_area   # [N1,N2]

    return giou[...,0]


if __name__ == '__main__':

    from keras.layers import Input, Lambda
    from keras.models import Model

    pred_cls = Input((3, 2))
    pred_bbox = Input((3, 4))
    tgt = Input((3, 2+4))

    # loss = detr_loss(pred, tgt, n_classes=2, cost_cls=1, cost_bbox=5, cost_giou=2)
    loss = Lambda(detr_loss, arguments={'n_classes': 2}, output_shape=detr_loss_output)([pred_cls,pred_bbox,tgt])
    print('loss', loss)
    model = Model([pred_cls,pred_bbox,tgt], loss)

    import numpy as np
    np.set_printoptions(threshold=np.inf)
    pred = np.array([[[1,0,0.2,0.2,0.1,0.1],    # [2,3,2+4]
                      [1,0,0.4,0.3,0.2,0.2],
                      [1,0,0.5,0.3,0.2,0.2]],
                     [[1,0,0.2,0.2,0.1,0.1],
                      [1,0,0.4,0.3,0.2,0.2],
                      [1,0,0.5,0.3,0.2,0.2]]])
    tgt = np.array([[[1,0,0.2,0.2,0.11,0.1],    # [2,3,2+4]
                     [1,0,0.4,0.3,0.2,0.2],
                     [0,1,0.5,0.3,0.2,0.2]],
                    [[1,0,0.2,0.2,0.1,0.1],
                     [0,1,0.4,0.3,0.2,0.2],
                     [0,1,0.5,0.3,0.2,0.2]]])
    loss = model.predict([pred[...,:2], pred[...,2:], tgt])
    print(loss)



