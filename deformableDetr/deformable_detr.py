from resnet import resnet
from GroupNormalization import GroupNormalization
from loss import detr_loss
from deformable_transformer import TransformerEncoderLayer, TransformerDecoderLayer
from keras.layers import Input, Conv2D, Reshape, Lambda, Embedding, Dense, Layer
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import math
import numpy as np


def deformable_detr(input_shape=(512,512,3), n_classes=80, depth=50, dilation=False, n_levels=4,
                    pe='sine', enc_layers=6, dec_layers=6, n_queries=300,
                    emb_dim=256, n_heads=8, ffn_dim=1024, drop_rate=0.1, n_points=4,
                    two_stage=False, two_stage_n_proposals=300, mode='test'):

    # inpt
    inpt = Input(input_shape)    # [b,h,w,c]

    # back
    r50 = resnet(input_shape, depth=depth, dilation=dilation)
    feats = r50(inpt)    # stage 2/3/4 outputs, list of [b,h,w,c]

    # prep for transformer
    strides = [8,16,32]    # channels = [512,1024,2048]
    neck_pes = []
    neck_feats = []
    feat_shapes = []
    level_start_idx = [0]
    for i in range(n_levels):
        stride = strides[i] if i<len(strides) else strides[-1]*2
        feat_h, feat_w = (math.ceil(input_shape[0]/stride), math.ceil(input_shape[1]/stride))  # same pooling
        feat_shapes.append((feat_h,feat_w))   # [h,w]

        feat = feats[i] if i<len(strides) else feats[-1]

        # positional embedding: constant sine + trainable level_emb
        pe = PositionalEmbeddingSine(emb_dim, (feat_h,feat_w))(feat)   # [b,h,w,d]
        # pe = Lambda(get_sine_pe_tf, arguments={'emb_dim': emb_dim, 'feature_shape': (feat_h,feat_w)})(feat)
        pe = Reshape((feat_h*feat_w,emb_dim))(pe)
        neck_pes.append(pe)

        # aligned feats
        if i<len(strides):
            # input_proj: 1x1 conv, xavier init
            feat = Conv2D(emb_dim, 1, strides=1, padding='same', name='input_proj_%d' % i)(feat)   # [b,h,w,d]
        else:
            # 3x3 s2 conv
            feat = Conv2D(emb_dim, 3, strides=2, padding='same', name='input_proj_%d' % i)(feat)
        feat = GroupNormalization(groups=32)(feat)   # [b,h,w,d]
        feat = Reshape((feat_h*feat_w,emb_dim))(feat)
        neck_feats.append(feat)

        # split idx
        if i!=n_levels-1:
            level_start_idx.append(level_start_idx[-1]+feat_h*feat_w)   # [0, 4096, 5120, 5376]

    print(neck_pes, neck_feats)

    # -------- transformer encoder --------
    x = Lambda(tf.concat, arguments={'axis': 1})(neck_feats)   # running feats, [b,Sum_hw,C]
    pos = Lambda(tf.concat, arguments={'axis': 1})(neck_pes)   # for each layer, [b,Sum_hw,C]
    ref = Lambda(get_reference_points, arguments={'spatial_shapes': feat_shapes})(x)  # [b,5440,4,2]
    print('----- encoder input', x, pos, ref)


    for i in range(enc_layers):
        x = TransformerEncoderLayer(emb_dim, ffn_dim, drop_rate, activation='relu', n_levels=n_levels, n_heads=n_heads,
                                    n_points=n_points, spatial_shapes=feat_shapes, level_start_idx=level_start_idx)([x,pos,ref])

    print('----- encoder output', x)

    # -------- transformer decoder --------
    encoder_feats = x
    box_indices = Lambda(lambda x: tf.tile(tf.expand_dims(tf.range(n_queries),axis=0),[tf.shape(x)[0],1]))(x)   # [b,n_queries]
    decoder_inpts = Embedding(n_queries, emb_dim*2)(box_indices)   # [b,Nq,2c], query & pe, learnable
    print(decoder_inpts, tf.split(decoder_inpts, axis=-1, num_or_size_splits=2))
    x, pe = Lambda(tf.split, arguments={'num_or_size_splits': 2, 'axis': -1})(decoder_inpts)  # [b,Nq,c]
    ref = Dense(2, activation='sigmoid')(pe)  # [b,Nq,2], box init positions
    print('----- decoder input', x, pe, ref, encoder_feats)

    for i in range(dec_layers):
        # intermediate results
        x = TransformerDecoderLayer(emb_dim, ffn_dim, drop_rate, activation='relu', n_levels=n_levels, n_heads=n_heads,
                                    n_points=n_points, spatial_shapes=feat_shapes, level_start_idx=level_start_idx)([x,pe,ref,encoder_feats])
    print('----- decoder output', x)

    # head: linear
    cls_output = Dense(n_classes, activation='softmax')(x)    # [b,L,n_cls]
    bbox_output = MLP(x, emb_dim, n_classes, 3)    # 3-layer-MLP, [b,L,4]

    if mode=='test':
        model = Model(inpt, [cls_output,bbox_output])
    else:
        gt = Input((n_queries,n_classes+4))    # [b,L,cls+4], cls: fg + bg
        loss = Lambda(detr_loss, arguments={'n_classes': n_classes})([cls_output,bbox_output,gt])
        model = Model([inpt,gt], loss)

    return model


def MLP(x, hidden_dim, output_dim, n_layers):

    # [fc-relu]-fc
    for i in range(n_layers):
        if i!=n_layers-1:
            x = Dense(hidden_dim, activation='relu')(x)
        else:
            x = Dense(output_dim)(x)
    return x


def get_reference_points(x, spatial_shapes):
    '''
    - spatial_shapes: list of (H_,W_), feature shapes
    --------
    - output: (1,Sum_Li,nL,2)
    '''
    reference_points_list = []
    for lvl, (H_,W_) in enumerate(spatial_shapes):
        center_y, center_x = tf.meshgrid(tf.linspace(0.5, H_-0.5, H_), tf.linspace(0.5, W_-0.5, W_))
        grids = tf.reshape(tf.stack([center_x/W_,center_y/H_], axis=2), (H_*W_,2))
        reference_points_list.append(grids)
    reference_points = tf.concat(reference_points_list, axis=0)  # [Sum_Li,2], merge across levels
    Sum_Li = reference_points.shape[0]
    reference_points = tf.reshape(reference_points, (1,Sum_Li,1,2))  # [1,Sum_Li,2]
    nL = len(spatial_shapes)
    reference_points = tf.tile(reference_points, [tf.shape(x)[0],1,nL,1])  # [b,Sum_Li,nL,2]
    return reference_points


class PositionalEmbeddingSine(Layer):

    def __init__(self, emb_dim, feature_shape, **kargs):
        super(PositionalEmbeddingSine, self).__init__(**kargs)
        self.emb_dim = emb_dim
        self.feature_shape = feature_shape
        self.level_embedding = self.add_weight(shape=(emb_dim,),
                                               initializer='random_uniform',
                                               name='lvl_embed')

    def call(self, x):
        pe = get_sine_pe(self.emb_dim, self.feature_shape)   # [1,h,w,D]
        pe = pe + self.level_embedding
        pe = tf.tile(pe, [tf.shape(x)[0],1,1,1])
        return pe

    def compute_output_shape(self, input_shape):
        return (None,) + self.feature_shape + (self.emb_dim,)


def get_sine_pe_tf(x, emb_dim, feature_shape, temp=10000, normalize=True, eps=1e-6):
    PE = get_sine_pe(emb_dim, feature_shape)    # [1,h,w,emb_dim]
    PE = tf.constant(PE, dtype='float32')
    PE = tf.tile(PE, [tf.shape(x)[0],1,1,1])

    return PE


def get_sine_pe(emb_dim, feature_shape, temp=10000, normalize=True, eps=1e-6):
    # feature_shape: (h,w)
    # returns: [1,h,w,emd_dim] constant embedding, without weights, not trainable
    assert emb_dim%2==0, 'illegal embedding dim'
    h, w = feature_shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))   # [h,w]
    grid_x, grid_y = grid_x+1, grid_y+1    # index start from 1
    if normalize:
        grid_x = grid_x / (w+eps) * 2 * math.pi
        grid_y = grid_y / (h+eps) * 2 * math.pi
    single_dim = np.arange(emb_dim//2)            # [half_dim,]
    single_dim = temp ** (4*(single_dim//2)/emb_dim)   # enlarge the unlinear range [1,1000]

    pe_x = np.tile(np.expand_dims(grid_x, axis=2), [1,1,emb_dim//2]) / single_dim   # [h,w,half_dim]
    pe_y = np.tile(np.expand_dims(grid_y, axis=2), [1,1,emb_dim//2]) / single_dim

    pe_x = np.stack([np.sin(pe_x[:,:,::2]), np.cos(pe_x[:,:,1::2])], axis=3).reshape((h,w,emb_dim//2))   # [h,w,half_dim]
    pe_y = np.stack([np.sin(pe_y[:,:,::2]), np.cos(pe_y[:,:,1::2])], axis=3).reshape((h,w,emb_dim//2))

    PE = np.concatenate([pe_y,pe_x], axis=2)    # [h,w,emb_dim]
    PE = np.expand_dims(PE, axis=0)    # [1,h,w,emb_dim]

    return PE


if __name__ == '__main__':

    model = deformable_detr(input_shape=(512,512,3), n_classes=80, depth=50, dilation=False, n_levels=4,
                            pe='sine', enc_layers=6, dec_layers=6, n_queries=300,
                            emb_dim=256, n_heads=8, ffn_dim=1024, drop_rate=0.1, n_points=4,
                            two_stage=False, two_stage_n_proposals=300, mode='test')
    model.summary()



