from keras.utils import Sequence
import cv2
from PIL import Image
import numpy as np
import random
import glob
import os


class dataSequence(Sequence):

    def __init__(self, img_dir, anno_dir, input_shape, n_classes, batch_size=2, max_boxes=100,
                 is_train=False, max_size=1200, imagenet_norm=False):

        self.img_files = [i for i in glob.glob(img_dir + '/*jpg')]
        self.label_files = [i.replace(img_dir, anno_dir).replace('jpg', 'txt') for i in self.img_files]
        print('num of %s samples: ' % ('train' if is_train else 'val'), len(self.img_files))

        self.indices = np.arange(len(self.img_files))

        self.input_shape = input_shape
        self.n_classes = n_classes    # coco_ids: 1-90
        self.batch_size = batch_size
        self.max_boxes = max_boxes

        self.is_train = is_train
        self.imagenet_norm = imagenet_norm

    def __len__(self):
        return len(self.img_files) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        x_batch, mask_batch, y_batch = self.batch_data_generator_v1(batch_indices)
        return [x_batch, mask_batch, y_batch], np.zeros((self.batch_size,))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def batch_data_generator_v1(self, batch_indices):

        img_batch = []        # [b,h,w,3]
        mask_batch = []       # [b,h,w], MSA mask
        boxes_batch = []      # [b,N,4+(cls+2)], normed xcycwh, [0,1,...,cls,cls+1], dummy+n_cls+bg
        input_shape = self.input_shape

        for sample_idx in batch_indices:
            # img: [H,W,3], RGB, [0,1], arr
            # boxes: [N,5], [cls_id,xc,yc,w,h], normed
            img, boxes = self.data_generator(sample_idx, self.is_train)
            # print('get_item', img.shape)

            h, w = img.shape[:2]
            pad_h, pad_w = input_shape[0]-h, input_shape[1]-w
            img = np.pad(img, [[0,pad_h],[0,pad_w],[0,0]], mode='constant')
            img_batch.append(img)

            mask = np.zeros(input_shape+(1,))
            mask[h:] = 1
            mask[:,w:] = 1
            mask_batch.append(mask)

            h_ratio, w_ratio = h/input_shape[0], w/input_shape[1]
            boxes = boxes * np.array([1,w_ratio,h_ratio,w_ratio,h_ratio])
            pad_gap = self.max_boxes - len(boxes)
            if pad_gap<0:
                indices = np.arange(len(boxes))
                np.random.shuffle(indices)
                boxes = boxes[indices[:self.max_boxes]]
            else:
                pad_content = np.tile(np.array([[-1.,0.,0.,0.,0.]]), [pad_gap,1])
                boxes = np.concatenate([boxes, pad_content], axis=0)

            cls_indices = np.int32(boxes[:,0])   # abs ids, (N,)
            labels = np.zeros((self.max_boxes, 1+self.n_classes+1))
            labels[np.arange(self.max_boxes), cls_indices] = 1
            boxes = np.concatenate([labels,boxes[:,-4:]], axis=1)   # [b,N,cls+4]
            boxes_batch.append(boxes)

        img_batch = np.stack(img_batch,0)       # [b,h,w,3]
        mask_batch = np.stack(mask_batch,0)     # [b,h,w]
        boxes_batch = np.stack(boxes_batch,0)   # [b,N,1+4]

        return img_batch, mask_batch, boxes_batch

    def data_generator(self, index, is_train, max_size=1333):
        # augment img & boxes: hflip, resize, resize-crop-resize, ImageNet-norm

        # load single img
        img = Image.open(self.img_files[index])    # RGB,0-255

        # load anno boxes
        boxes = np.zeros((0, 5), dtype=np.float32)
        if os.path.isfile(self.label_files[index]):
            with open(self.label_files[index], 'r') as f:
                # [N,5], [cls_id,xc,yc,w,h], normed
                boxes = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

        if is_train:
            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

            # horizontal flip
            if random.uniform(0, 1)>0.5:
                img = np.flip(np.array(img), axis=1)
                img = Image.fromarray(img)
                if boxes.size>0:
                    boxes[:,[1]] = 1 - boxes[:,[1]]   # xc

            # resize
            if random.uniform(0, 1)>0.5:
                # resize
                size = random.choice(scales)
                img, boxes = resize(img, boxes, size, max_size=self.input_shape)

            else:
                # resize-crop-resize
                size = random.choice([400,500,600])
                img, boxes = resize(img, boxes, size, max_size=self.input_shape)
                img, boxes = crop(img, boxes, 384, 600)
                size = random.choice(scales)
                img, boxes = resize(img, boxes, size, max_size=self.input_shape)

        else:
            scales = [800]

            # resize
            size = random.choice(scales)
            img, boxes = resize(img, boxes, size, max_size=self.input_shape)

        # norm
        if self.imagenet_norm:
            ImageNet_mean = np.array([0.485, 0.456, 0.406])
            ImageNet_std = np.array([0.229, 0.224, 0.225])
            img = (np.array(img) - ImageNet_mean) / ImageNet_std
        else:
            img = np.array(img) / 255.

        return img, boxes   # pil_img


def resize(img, boxes, size, max_size):
    # keep ratio resize
    w, h = img.size
    ratio = min(size/min(w,h), max_size[0]/h, max_size[1]/w)
    new_w, new_h = int(w*ratio), int(h*ratio)
    img = img.resize((new_w, new_h))

    return img, boxes


def crop(img, boxes, min_size, max_size):

    w, h = img.size
    crop_w = random.randint(min_size, min(w, max_size))
    crop_h = random.randint(min_size, min(h, max_size))
    x0 = random.randint(0, w-crop_w)
    y0 = random.randint(0, h-crop_h)

    # crop img
    img = np.array(img)
    crop_img = img[y0:y0+crop_h, x0:x0+crop_w]
    crop_img = Image.fromarray(crop_img)

    # crop boxes, xcycwh
    boxes_wh = boxes[:,3:]    # normed_wh
    boxes[:,1:3] = boxes[:,1:3] - boxes_wh/2.   # x1y1
    boxes[:,3:] = boxes[:,3:] + boxes[:,1:3]   # x2y2
    boxes_abs = boxes * np.array([1,w,h,w,h])   # cls_id + abs_x1y1x2y2

    boxes_abs[:,1] = np.maximum(boxes_abs[:,1],x0)
    boxes_abs[:,2] = np.maximum(boxes_abs[:,2],y0)
    boxes_abs[:,3] = np.minimum(boxes_abs[:,3],x0+crop_w)
    boxes_abs[:,4] = np.minimum(boxes_abs[:,4],y0+crop_h)
    boxes_ratio = np.maximum((boxes_abs[:,3]-boxes_abs[:,1])/(boxes_abs[:,4]-boxes_abs[:,2]),
                             (boxes_abs[:,4]-boxes_abs[:,2])/(boxes_abs[:,3]-boxes_abs[:,1]))
    # print('before', boxes_abs.shape)

    valid = (boxes_abs[:,1]<boxes_abs[:,3]) & (boxes_abs[:,2]<boxes_abs[:,4]) & (boxes_ratio<10)
    valid_indices = np.where(valid)
    boxes_abs = boxes_abs[valid_indices]
    # print('after', boxes_abs.shape)

    # normed boxes
    boxes_xcyc = ((boxes_abs[:,1:3] + boxes_abs[:,3:]) / 2 - np.array([x0,y0])) / np.array([crop_w,crop_h])
    boxes_wh = (boxes_abs[:,3:] - boxes_abs[:,1:3]) / np.array([crop_w,crop_h])
    boxes = np.concatenate([boxes_abs[:,:1],boxes_xcyc,boxes_wh],axis=1)

    return crop_img, boxes


if __name__ == '__main__':

    img_dir = "coco/train/"
    anno_dir = "coco/txt/"
    input_shape = (800,1200)
    n_classes = 90
    batch_size = 2
    max_boxes = 10

    data_generator = dataSequence(img_dir, anno_dir, input_shape, n_classes, batch_size,
                                  max_boxes=max_boxes, is_train=True, max_size=1200)

    for idx, [[img_batch, mask_batch, box_batch],_] in enumerate(data_generator):

        # # v1: raw boxes
        print(img_batch.shape, mask_batch.shape, box_batch.shape)
        batch_size = len(img_batch)
        for i in range(batch_size):

            canvas = img_batch[i][:,:,::-1].copy()  # BGR
            mask = mask_batch[i]  # (h,w,1)
            canvas_h, canvas_w = canvas.shape[:2]
            obj_classes = []
            for b in box_batch[i]:
                xc, yc, w, h = b[-4:]
                cls_id = np.argmax(b[:-4])
                if cls_id<=n_classes:
                    obj_classes.append(cls_id)
                cv2.rectangle(canvas, (int((xc-w/2.)*canvas_w), int((yc-h/2.)*canvas_h)),
                              (int((xc+w/2.)*canvas_w), int((yc+h/2.)*canvas_h)),
                              (0,0,255), 2)
            mask = np.pad(mask, [[0,0],[0,0],[0,2]])
            canvas = cv2.addWeighted(canvas, 0.6, mask, 0.4, 0.)
            print(obj_classes)
            cv2.imshow('tmp', canvas)
            cv2.waitKey(0)

        if idx>10:
            break















