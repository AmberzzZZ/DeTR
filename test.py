import numpy as np
from PIL import Image, ImageDraw
import os
from detr import detr


if __name__ == '__main__':

    data_dir = "coco/val/tinyval2014/"

    # # padded data
    # max_h, max_w = 0, 0
    # img_lst = []
    # for file in os.listdir(data_dir):
    #     img = Image.open(os.path.join(data_dir, file))
    #     w, h = img.size
    #     ratio = min(1333/max(w,h), 800/min(w,h))
    #     new_w, new_h = int(w*ratio), int(h*ratio)
    #     img = np.array(img.resize((new_w,new_h))) / 255.
    #     img_lst.append(img)
    #     max_h = max(max_h, new_h)
    #     max_w = max(max_w, new_w)
    # print('max_h', max_h, 'max_w', max_w)

    # batch_lst = []
    # for img in img_lst:
    #     h, w, c = img.shape
    #     pad_h, pad_w = max_h-h, max_w-w
    #     img = np.pad(img, [[0,pad_h],[0,pad_w],[0,0]], mode='constant')
    #     batch_lst.append(img)
    # batch_inpt = np.stack(batch_lst, axis=0)   # [b,h,w,c]

    # model
    model = detr(input_shape=(800,1200,3), pe='sine', n_classes=92, depth=50, dilation=False,
                 emb_dim=256, enc_layers=6, dec_layers=6, max_boxes=100)
    model.load_weights("weights/detr-r50.h5")

    # run prediction
    for file in os.listdir(data_dir):
        img = Image.open(os.path.join(data_dir,file))

        inpt = np.array(img.resize((1200,800)))
        inpt = np.expand_dims(inpt/255., axis=0)   # [1,h,w,3]

        preds = model.predict(inpt)
        cls_preds = preds[0][0,:,:-1]    # fg scores, [N,cls]
        box_preds = preds[1][0]   # [N,4]

        scores = np.max(cls_preds, axis=-1)    # [N,]
        labels = np.argmax(cls_preds, axis=-1)  # [N,]

        boxes_xcycwh = box_preds  # [N,4], 0-1 normed, xcycwh wrt. origin wh
        boxes_x1y1x2y2 = np.stack([boxes_xcycwh[:,0]-boxes_xcycwh[:,2]/2,   # [N,4], normed x1y1x2y2
                                   boxes_xcycwh[:,1]-boxes_xcycwh[:,3]/2,
                                   boxes_xcycwh[:,0]+boxes_xcycwh[:,2]/2,
                                   boxes_xcycwh[:,1]+boxes_xcycwh[:,3]/2], axis=1)

        # draw fg boxes
        N = scores.shape[0]
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(N):
            if scores[i]>0.5:
                x1,y1,x2,y2 = boxes_x1y1x2y2[i]*np.array([w,h,w,h]).tolist()
                draw.rectangle((x1,y1,x2,y2), fill=None, outline=(0,0,0), width=5)
        img.save('preds/'+file)














