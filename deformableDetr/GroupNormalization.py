from keras.engine import Layer
from keras import initializers, regularizers, constraints
import keras.backend as K
import tensorflow as tf


class GroupNormalization(Layer):
    """Group normalization layer
    # input shape: [b,h,w,c]
    # groups: num of groups splitting the channel
    # output shape: Same shape as input.
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-5,
                 center=True,     # reshift
                 scale=True,      # rescale
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):

        dim = input_shape[self.axis]
        assert dim % self.groups == 0, 'channel/groups value error'

        if self.scale:
            self.gamma = self.add_weight(shape=(dim,),
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)

        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=(dim,),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)

        else:
            self.beta = None

        self.built = True

    def call(self, x, training=None, **kwargs):

        _, h, w, c = x.shape
        x = tf.reshape(x, (-1,h,w,self.groups,c//self.groups))
        x = tf.transpose(x, (0,3,1,2,4))   # [b,G,h,w,c/G]

        # norm
        g_mean = K.mean(x, axis=[2,3,4], keepdims=True)
        g_var = K.var(x, axis=[2,3,4], keepdims=True)
        x = (x - g_mean) / (g_var + self.epsilon)   # [b,]

        # rescale
        x = tf.reshape(tf.transpose(x,(0,2,3,1,4)), (-1,h,w,c))
        x = x * self.gamma + self.beta

        return x

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':

    from keras.layers import Input

    x = Input((96,128,128,32))
    x = Input((128,128,32))
    y = GroupNormalization(axis=-1, groups=4)(x)
    print(y)

    layer = GroupNormalization(axis=-1, groups=4)
    y = layer(y)
    print(layer.weights)





