from resnet import resnet
from MSA import MultiHeadAttention, FeedForwardNetwork
from LayerNormalization import LayerNormalization
from loss import detr_loss, detr_loss_output
from keras.layers import Input, Conv2D, Embedding, Dropout, add, Reshape, Dense, Lambda, ReLU
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import numpy as np
import math


def detr(input_shape=(512,512,3), pe='sine', n_classes=80, depth=50, dilation=False,
         emb_dim=256, enc_layers=6, dec_layers=6, max_boxes=100, max_target_boxes=20,
         mlp_dim=256, mode='test'):

    # inpt
    inpt = Input(input_shape)    # [b,h,w,c]
    mask = Input((input_shape[0],input_shape[1],1))   # [b,h,w,1]

    # back
    r50 = resnet(input_shape, depth=depth, dilation=dilation)
    feat = r50(inpt)    # stage 5 output
    stride = 32 // (2**int(dilation))
    feature_shape = (math.ceil(input_shape[0]/stride), math.ceil(input_shape[1]/stride))
    print('feature shape', feature_shape)
    feat_mask = Lambda(lambda x: tf.image.resize(x, size=feature_shape))(mask)   # interpolate

    # reflect & pe
    x = Conv2D(emb_dim, 1, strides=1, padding='same', name='input_proj')(feat)   # [b,h,w,d]
    x = Reshape((feature_shape[0]*feature_shape[1], emb_dim))(x)                 # [b,hw,d]
    if pe=='sine':   # constant
        # feat_pe = PositionalEmbeddingSine(emb_dim, feature_shape)  # [1,h,w,d]
        feat_pe = Lambda(lambda x: PositionalEmbeddingSine(emb_dim, feature_shape), name='PESine')(x)
    elif pe=='learned':   # trainable weights
        feat_pe = PositionalEmbeddingLearned(emb_dim, feature_shape, name='PELearned')(x)
    feat_pe = Reshape((feature_shape[0]*feature_shape[1], emb_dim))(feat_pe)    # [1,hw,d]
    feat_mask = Reshape((feature_shape[0]*feature_shape[1],))(feat_mask)        # [b,hw]
    print('transformer feat', feat_pe, feat_mask)

    # transformer encoder: parse inputs
    for i in range(enc_layers):
        x = TransformerEncoderBlock(drop_rate=0.1)([x, feat_pe], mask=feat_mask)      # [b,hw,d]
    encoder_feats = x

    # transformer decoder: feed targets x is a zeros-variable initially, get updated through the decoder blocks
    x, target_pe = PrepareDecoderInput(max_boxes, emb_dim)(x)
    print('decoder pe', target_pe)

    for i in range(dec_layers):
        x = TransformerDecoderBlock(drop_rate=0.1)([x, target_pe, encoder_feats, feat_pe], key_mask=feat_mask)
    x = LayerNormalization()(x)    # norm no matter pre_norm/post_norm

    # head: mlp
    cls_output = Dense(n_classes, name='cls_pred', activation='softmax')(x)  # [b,N2,cls]
    box = Dense(mlp_dim, activation='relu', name='box_hidden_1')(x)
    box = Dense(mlp_dim, activation='relu', name='box_hidden_2')(box)
    box_output = Dense(4, activation='sigmoid', name='box_pred')(box)    # [b,N2,4]

    if mode=='test':
        model = Model([inpt,mask], [cls_output,box_output])
    else:
        gt = Input((max_target_boxes,n_classes+4))    # [b,N2,4], fg + bg
        loss = Lambda(detr_loss, arguments={'n_classes': n_classes})(
                      [cls_output,box_output,gt])
        model = Model([inpt,mask,gt], loss)

    return model


class PrepareDecoderInput(Model):
    def __init__(self, max_boxes=100, emb_dim=256, name='query_embed'):
        super(PrepareDecoderInput, self).__init__(name=name)
        self.emb = Embedding(max_boxes, emb_dim)
        self.max_boxes = max_boxes
        self.emb_dim = emb_dim

        # create variables
        self.target_indices = tf.expand_dims(tf.range(self.max_boxes), axis=0)  # [1,N1]
        self.target = tf.zeros((1, max_boxes, emb_dim))     # [1,N1,d], zeros

    def call(self, x):
        # x: encoder feats: [b,hw,d]
        b = tf.shape(x)[0]
        target_indices = tf.tile(self.target_indices, [b,1])    # [b,N1]
        target_pe = self.emb(target_indices)     # [b,N1,d], take corresponding word-vecs
        target = tf.tile(self.target, [b,1,1])   # [b,N1,d]
        return [target, target_pe]

    def compute_output_shape(self, input_shape):
        b, N, d = input_shape
        return [(b,self.max_boxes,d), (b,self.max_boxes,d)]


class TransformerEncoderBlock(Model):
    def __init__(self, attn_dim=256, ffn_dim=2048, drop_rate=0.1, norm_before=False):
        super(TransformerEncoderBlock, self).__init__()

        self.ln1 = LayerNormalization()
        self.msa = MultiHeadAttention(num_heads=8, model_size=attn_dim, name='self_att')
        self.drop1 = Dropout(drop_rate)

        self.ln2 = LayerNormalization()
        self.dense1 = Dense(ffn_dim)
        self.act1 = ReLU()
        self.dense2 = Dense(attn_dim)
        self.drop2 = Dropout(drop_rate)
        self.drop3 = Dropout(drop_rate)

        self.norm_before = norm_before

    def call(self, inputs, mask=None):
        if self.norm_before:
            return self.pre_norm(inputs, mask)
        else:
            return self.post_norm(inputs, mask)

    def pre_norm(self, inputs, mask=None):
        x, pe = inputs

        # id path
        inpt = x

        # residual path
        x = self.ln1(x)
        q = k = x + pe
        v = x
        x = self.msa([q,k,v], mask=mask)
        x = self.drop1(x)

        # add
        x = x + inpt

        # id path
        ffn_inpt = x

        # residual path
        x = self.ln2(x)
        x = self.drop2(self.act1(self.dense1(x)))
        x = self.drop3(self.dense2(x))

        # add
        x = x + ffn_inpt

        return x

    def post_norm(self, inputs, mask=None):
        x, pe = inputs

        # id path
        inpt = x

        # residual path
        q = k = x + pe
        v = x
        x = self.msa([q,k,v], mask=mask)
        x = self.drop1(x)

        # add
        x = x + inpt
        x = self.ln1(x)

        # id path
        ffn_inpt = x

        # residual path
        x = self.drop2(self.act1(self.dense1(x)))
        x = self.drop3(self.dense2(x))

        # add
        x = x + ffn_inpt
        x = self.ln2(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class TransformerDecoderBlock(Model):
    def __init__(self, attn_dim=256, ffn_dim=2048, drop_rate=0.1, norm_before=False):
        super(TransformerDecoderBlock, self).__init__()

        self.ln1 = LayerNormalization()
        self.msa1 = MultiHeadAttention(num_heads=8, model_size=attn_dim, name='self_att')   # self-att
        self.drop1 = Dropout(drop_rate)

        self.ln2 = LayerNormalization()
        self.msa2 = MultiHeadAttention(num_heads=8, model_size=attn_dim, name='mutual_att')   # mutual-att
        self.drop2 = Dropout(drop_rate)

        self.ln3 = LayerNormalization()
        self.dense1 = Dense(ffn_dim)
        self.act1 = ReLU()
        self.dense2 = Dense(attn_dim)
        self.drop3 = Dropout(drop_rate)
        self.drop4 = Dropout(drop_rate)

        self.norm_before = norm_before

    def call(self, inputs, mask=None, key_mask=None):
        # targets: decoder input
        # inputs: encoder output
        if self.norm_before:
            return self.pre_norm(inputs, key_mask=key_mask)
        else:
            return self.post_norm(inputs, key_mask=key_mask)

    def pre_norm(self, inputs, key_mask=None):
        x, target_pe, inputs, feat_pe = inputs

        # id path
        inpt = x

        # residual path
        x = self.ln1(x)
        q = k = x + target_pe
        v = x
        x = self.msa1([q,k,v])
        x = self.drop1(x)

        # add
        x = x + inpt

        # id path
        inpt = x

        # residual path
        x = self.ln2(x)
        q = x + target_pe
        k = inputs + feat_pe
        v = inputs
        x = self.msa2([q,k,v], key_mask=key_mask)
        x = self.drop2(x)

        # add
        x = x + inpt

        # id path
        ffn_inpt = x

        # residual path
        x = self.ln3(x)
        x = self.drop3(self.act1(self.dense1(x)))
        x = self.drop4(self.dense2(x))

        # add
        x = x + ffn_inpt

        return x

    def post_norm(self, inputs, key_mask=None):
        x, target_pe, inputs, feat_pe = inputs

        # id path
        inpt = x

        # residual path
        q = k = x + target_pe
        v = x
        x = self.msa1([q,k,v])
        x = self.drop1(x)

        # add
        x = x + inpt
        x = self.ln1(x)

        # id path
        inpt = x

        # residual path
        q = x + target_pe
        k = inputs + feat_pe
        v = inputs
        x = self.msa2([q,k,v], key_mask=key_mask)
        x = self.drop2(x)

        # add
        x = x + inpt
        x = self.ln2(x)

        # id path
        ffn_inpt = x

        # residual path
        x = self.drop3(self.act1(self.dense1(x)))
        x = self.drop4(self.dense2(x))

        # add
        x = x + ffn_inpt
        x = self.ln3(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def PositionalEmbeddingSine(emb_dim, feature_shape, temp=10000, normalize=True, eps=1e-6):
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
    PE = K.constant(np.expand_dims(PE, axis=0))    # [1,h,w,emb_dim]

    return PE


class PositionalEmbeddingLearned(Model):
    # returns: [1,h,w,emd_dim] learnable embedding, with weights, trainable
    def __init__(self, emb_dim, feature_shape, name=None):
        super(PositionalEmbeddingLearned, self).__init__(name=name)
        assert emb_dim%2==0, 'illegal embedding dim'
        self.emb_x = Embedding(50, emb_dim//2)
        self.emb_y = Embedding(50, emb_dim//2)
        self.emb_dim = emb_dim
        self.feature_shape = feature_shape

    def call(self, inputs):
        h, w = self.feature_shape
        coord_x = tf.constant(np.arange(w).reshape((1,w)))   # constant, [1,w]
        coord_y = tf.constant(np.arange(h).reshape((1,h)))

        emb_x = self.emb_x(coord_x)   # [1,w,half_dim]
        emb_y = self.emb_y(coord_y)   # [1,h,half_dim]

        pe_x = tf.tile(emb_x, [h,1,1])
        pe_y = tf.tile(tf.transpose(emb_y, (1,0,2)), [1,w,1])

        PE = tf.concat([pe_x,pe_y], axis=2)   # [h,w,emb_dim]
        PE = tf.expand_dims(PE, axis=0)       # [1,h,w,emb_dim]

        return PE

    def compute_output_shape(self, input_shape):
        h,w = self.feature_shape
        return (None,h,w,self.emb_dim)


if __name__ == '__main__':

    pe = PositionalEmbeddingSine(256, (25,38), temp=10000, normalize=True, eps=1e-6)
    print(pe)

    pe_layer = PositionalEmbeddingLearned(128, (8,10))
    x = tf.ones((32,32))
    y = pe_layer(x)
    print(pe_layer.weights)  # The first call will create the weights

    model = detr(input_shape=(512,512,3), pe='sine', n_classes=92, depth=50, dilation=False,
                 emb_dim=256, enc_layers=6, dec_layers=6, max_boxes=100,
                 mode='train')
    model.summary()
    model.load_weights("weights/detr-r50.h5")

    # for l in model.layers:
    #     if 'decoderblock_1' in l.name:
    #         print(l.weights)




