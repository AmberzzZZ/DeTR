from keras.models import Model
from keras.layers import Dense
import tensorflow as tf


class MSDeformAttention(Model):
    def __init__(self, model_dim, n_heads=8, n_levels=4, n_points=4, spatial_shapes=None, level_start_idx=None):
        super(MSDeformAttention, self).__init__()
        self.value_proj = Dense(model_dim)
        self.samping_offsets = Dense(n_heads*n_levels*n_points*2)
        self.attention_weights = Dense(n_heads*n_levels*n_points, activation='softmax')
        self.output_proj = Dense(model_dim)

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.spatial_shapes = spatial_shapes

    def call(self, inputs, mask=None):
        '''
        - query: [b,Lq,D], feat+pe
        - value: [b,Lv,D], feat, Lv=Sum_Li, num of feature grids, former outputs
        - mask: mask on value

        - reference points: [b,Lq,nL,2], normed grid centers, [0,1]
        - spatial_shapes: list of (H_,W_)
        - level_start_idx: [nL]
        '''

        query, value, reference_points = inputs

        bs = tf.shape(query)[0]  # tensor
        len_q = int(query.shape[1])
        len_v = int(value.shape[1])

        # linear projection
        value = self.value_proj(value)  # [b,Lv,C]
        if mask is not None:
            value = value * mask   # block the invalid edges
        # split head
        value = tf.reshape(value, (bs,len_v, self.n_heads, self.model_dim//self.n_heads))  # [b,Lv,nH,C/nH]

        # offsets: dense on query
        sampling_offsets = self.samping_offsets(query)  # [b,Lq,nH*nL*nP*2], unlimited
        sampling_offsets = tf.reshape(sampling_offsets, (bs,len_q,self.n_heads,self.n_levels,self.n_points,2))
        # coords
        norm_factor = tf.stack(self.spatial_shapes)  # [nL,2]
        norm_factor = tf.cast(norm_factor, tf.float32)
        norm_factor = tf.reshape(norm_factor, (1,1,1,self.n_levels,1,2))
        reference_points = tf.reshape(reference_points, (bs,len_q,1,self.n_levels,1,2))
        sampling_coords = reference_points + sampling_offsets/norm_factor   # [b,Lq,nH,nL,nP,2]

        # attention: dense on query
        attention_weights = self.attention_weights(query)  # [b,Lq,nH*nL*nP]
        attention_weights = tf.reshape(attention_weights, (bs,len_q,self.n_heads,self.n_levels*self.n_points))
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = tf.reshape(attention_weights, (bs,len_q,self.n_heads,self.n_levels, self.n_points))  # [b,Lq,nH,nL,nP]

        # split levels
        value_lst = tf.split(value, axis=1, num_or_size_splits=[h*w for h,w in self.spatial_shapes])
        # compute attention
        sampling_lst = []
        for lvl, (H_,W_) in enumerate(self.spatial_shapes):
            value_l = value_lst[lvl]    # [b,HW,nH,C]
            value_l = tf.transpose(value_l, (0,2,1,3))
            value_l = tf.reshape(value_l, (bs*self.n_heads,H_,W_,self.model_dim//self.n_heads))  # [b*nH,H,W,C]
            sampling_coords_l = sampling_coords[:,:,:,lvl]   # [b,Lq,nH,nP,2]
            sampling_coords_l = tf.transpose(sampling_coords_l, (0,2,1,3,4))    # [b,nH,Lq,nP,2]
            sampling_coords_l = tf.reshape(sampling_coords_l, (bs*self.n_heads,len_q,self.n_points,2))  # [b*nH,Lq,nP,2]
            # bilinear_sampler
            sampled_feats = bilinear_sampler(value_l, sampling_coords_l)  # [b*nH,Lq,nP,C]
            sampling_lst.append(sampled_feats)

        sampling_lst = tf.stack(sampling_lst, axis=2)  # [b*nH,Lq,nL,nP,C]
        sampling_lst = tf.reshape(sampling_lst, (bs*self.n_heads,len_q,self.n_levels*self.n_points,self.model_dim//self.n_heads))   # [b*nH,Lq,nL*nP,C]
        attention_weights = tf.transpose(attention_weights, (0,2,1,3,4))  # [b,nH,Lq,nL,nP]
        attention_weights = tf.reshape(attention_weights, (bs*self.n_heads,len_q,self.n_levels*self.n_points,1))   # [b*nH,Lq,nL*nP,1]
        # matmul on last dim
        output = tf.matmul(attention_weights, sampling_lst, transpose_a=True)  # [b*nH,Lq,C/nH]
        output = tf.reshape(output, (bs,self.n_heads,len_q,self.model_dim//self.n_heads))
        output = tf.transpose(output, (0,2,1,3))
        output = tf.reshape(output, (bs, len_q, self.model_dim))

        # fuse multi heads
        output = self.output_proj(output)

        return output

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[0]
        return (B,N,self.model_dim)


def bilinear_sampler(img, grids):
    """
    - img: batch of images in (B, H, W, C) layout
    - grid_xy: (B,out_H,out_W,2), [-1,1]
    -------
    - output: interpolated images according to grids, (B,out_H,out_W,C)
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x, y = tf.split(grids, 2, axis=-1)
    x = tf.squeeze(x)
    y = tf.squeeze(y)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)  # [b,h,w]
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


def get_pixel_value(img, x, y):
    """
    - img: tensor of shape (B, H, W, C)
    - x: tensor of shape (B,h,w,)
    - y: tensor of shape (B,h,w,)
    -------
    - output: tensor of shape (B, h, w, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)  # [b,h,w,3]

    return tf.gather_nd(img, indices)


if __name__ == '__main__':

    from keras.layers import Input

    spatial_shapes = [(64,64),(32,32),(16,16),(8,8)]
    level_start_idx = [0, 4096, 5120, 5376]

    q = Input((5440,256))  # [b,L,D]
    v = Input((5440,256))  # [b,L,D]
    ref = Input((5440,4,2))
    layer = MSDeformAttention(model_dim=256, n_heads=8, n_levels=4, n_points=4,
                              spatial_shapes=spatial_shapes, level_start_idx=level_start_idx)

    y = layer([q,v,ref])
    print(y)


    model = Model([q,v,ref], y)

    import numpy as np
    q = np.random.uniform(0, 1, (2,5440,256))
    v = np.random.uniform(0, 1, (2,5440,256))
    ref = np.random.uniform(0, 1, (2,5440,4,2))
    y = model.predict([q,v,ref])
    print(y.shape)





