from keras.models import Model
from keras.layers import Dropout, Dense, Activation
from MSA import MultiHeadAttention
from MSDeformAttn import MSDeformAttention
from LayerNormalization import LayerNormalization
import tensorflow as tf
import keras.backend as K


class TransformerEncoderLayer(Model):

    def __init__(self, model_dim=256, ffn_dim=1024, drop_rate=0.1, activation='relu',
                 n_levels=4, n_heads=8, n_points=4, spatial_shapes=[], level_start_idx=[]):
        super(TransformerEncoderLayer, self).__init__()

        # self-att
        self.self_att = MSDeformAttention(model_dim, n_heads, n_levels, n_points, spatial_shapes,
                                          level_start_idx)
        self.drop1 = Dropout(drop_rate)
        self.ln1 = LayerNormalization()

        # ffn
        self.dense1 = Dense(ffn_dim)
        self.act = Activation(activation)
        self.drop2 = Dropout(drop_rate)
        self.dense2 = Dense(model_dim)
        self.drop3 = Dropout(drop_rate)
        self.ln2 = LayerNormalization()

        # hyper
        self.model_dim = model_dim

    def call(self, inputs):
        '''
        - src: flattened features, [b,Sum_Li,C], multi levels
        - pos: positional embeddings, [b,Sum_Li,C]
        # - mask: flattened mask, [b,Sum_Li]

        - reference_points: (b,Sum_Li,nL,2), grid coords, across levels
        - spatial_shapes: list of (H_,W_), feature shapes
        - level_start_idx: [nL], split indices
        '''

        src, pos, reference_points = inputs

        # att
        inpt = src
        x = self.self_att([src+pos, src, reference_points])   # query,value,ref
        x = inpt + self.drop1(x)
        x = self.ln1(x)

        # ffn
        inpt = x
        x = self.dense2(self.drop2(self.act(self.dense1(x))))
        x = inpt + self.drop3(x)
        x = self.ln2(x)

        return x

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[0]
        return (B,N,self.model_dim)


class TransformerDecoderLayer(Model):

    def __init__(self, model_dim=256, ffn_dim=1024, drop_rate=0.1, activation='relu',
                 n_levels=4, n_heads=8, n_points=4, spatial_shapes=[], level_start_idx=[]):
        super(TransformerDecoderLayer, self).__init__()

        # self-att: running query & running query
        self.self_att = MultiHeadAttention(model_dim, n_heads, drop_rate, drop_rate)
        self.drop1 = Dropout(drop_rate)
        self.ln1 = LayerNormalization()

        # cross-att: running query & encoder output
        self.cross_att = MSDeformAttention(model_dim, n_heads, n_levels, n_points, spatial_shapes,
                                           level_start_idx)
        self.drop2 = Dropout(drop_rate)
        self.ln2 = LayerNormalization()

        # ffn
        self.dense1 = Dense(ffn_dim)
        self.act = Activation(activation)
        self.drop3 = Dropout(drop_rate)
        self.dense2 = Dense(model_dim)
        self.drop4 = Dropout(drop_rate)
        self.ln3 = LayerNormalization()

        # hyper
        self.model_dim = model_dim

    def call(self, inputs):
        '''
        - x: query, [b,n_boxes,C]
        - pos: query pe,
        - ref: reference_points of the query, grid centers in encoder/ in decoder proposals
        - encoder_feats: [b,Sum_Li,C], value
        '''

        x, pos, ref, encoder_feats = inputs

        # self-att
        inpt = x
        x = self.self_att([x+pos,x+pos, x])  # q=k=x+pe, v=x
        x = inpt + self.drop1(x)
        x = self.ln1(x)

        # cross-att
        inpt = x
        x = self.cross_att([x+pos, encoder_feats, ref])  # q=x+pe, v=value, ref
        x = inpt + self.drop2(x)
        x = self.ln2(x)

        # ffn
        inpt = x
        x = self.dense2(self.drop2(self.act(self.dense1(x))))
        x = inpt + self.drop3(x)
        x = self.ln2(x)

        return x

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[0]
        return (B,N,self.model_dim)


if __name__ == '__main__':

    from keras.layers import Input

    spatial_shapes = [(64,64),(32,32),(16,16),(8,8)]
    level_start_idx = [0, 4096, 5120, 5376]
    for a,b in spatial_shapes:
        if not level_start_idx:
            level_start_idx.append(a*b-1)
        else:
            level_start_idx.append(level_start_idx[-1]+a*b-1)

    x = Input((5440,256))  # [b,L,D]
    pe = Input((5440,256))
    ref = Input((5440,4,2))
    layer = TransformerEncoderLayer(model_dim=256, ffn_dim=1024, drop_rate=0.1, activation='relu',
                                    n_levels=4, n_heads=8, n_points=4,
                                    spatial_shapes=spatial_shapes, level_start_idx=level_start_idx)
    y = layer([x,pe,ref])
    print(y)






