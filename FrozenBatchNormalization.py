from keras.layers import Layer
import keras.backend as K


class FrozenBatchNormalization(Layer):

    # given inputs: [b,(hwd),c], compute norm over the (bhwd)-dim

    def __init__(self,
                 epsilon=1e-5,
                 **kwargs):
        super(FrozenBatchNormalization, self).__init__(**kwargs)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon

    def build(self, input_shape):
        # rescale factor, for each sample, broadcast from last-dim
        self.n_dim = len(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(
            shape=shape,
            initializer='ones',
            name='gamma',
            trainable=False,
        )
        self.beta = self.add_weight(
            shape=shape,
            initializer='zeros',
            name='beta',
            trainable=False,
        )
        self.mean = self.add_weight(
            shape=shape,
            initializer='zeros',
            name='mean',
            trainable=False,
        )
        self.variance = self.add_weight(
            shape=shape,
            initializer='ones',
            name='variance',
            trainable=False,
        )

    def call(self, inputs, training=None):
        # norm
        outputs = (inputs - self.mean) / K.sqrt(self.variance + self.epsilon)
        # rescale
        outputs = self.gamma*outputs + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':

    from keras.layers import Input
    from keras.models import Model

    x = Input((2,32,32,128))
    y = FrozenBatchNormalization()(x)

    model = Model(x,y)
    model.summary()

    print(y)

