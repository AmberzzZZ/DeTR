from detr import detr
# from AdamW import AdamW
from dataSequence import dataSequence
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam


if __name__ == '__main__':

    # data
    img_dir = "../coco/train/"
    anno_dir = "../coco/txt/"
    input_shape = (800,1200)
    n_classes = 90
    batch_size = 2
    max_target_boxes = 20

    train_generator = dataSequence(img_dir, anno_dir, input_shape, n_classes, batch_size,
                                   max_boxes=max_target_boxes, is_train=True, max_size=1200)

    # model
    model = detr(input_shape=input_shape+(3,), n_classes=92, depth=50, dilation=False,
                 pe='sine', emb_dim=256, enc_layers=6, dec_layers=6,
                 max_boxes=100, max_target_boxes=max_target_boxes, mode='train')
    model.load_weights("weights/detr-r50.h5")
    for sub_l in model.get_layer('backbone').layers:
        if 'stem' in sub_l.name or 'bn' in sub_l.name:
            print('freeze layer: backbone/', sub_l.name)
            sub_l.trainable = False

    # opt, lr schedule
    # adamW, clip_norm=0.1
    # lr: backbone: 1e-5, rest: 1e-4, torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1)
    # weight decay: 1e-4

    model.compile(Adam(1e-5), loss=lambda y_true,y_pred: y_pred)

    model.fit_generator(train_generator,
                        steps_per_epoch=2,
                        epochs=2,
                        verbose=1,
                        callbacks=None,
                        initial_epoch=0)



