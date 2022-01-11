# resnet 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, GlobalAveragePooling2D, \
                         Reshape, Dense, multiply
from keras.models import Model
from FrozenBatchNormalization import FrozenBatchNormalization


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
n_filters = [256, 512, 1024, 2048]


def resnet(input_shape=(224,224,3), depth=50, dilation=[False,False,False]):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    for i in range(len(num_blocks)):
        dilate = dilation[i-1]
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 else 1
            if dilate:
                dilate *= strides
                strides = 1
            else:
                dilate = 1
            if j==0:
                x = res_block(x, n_filters[i], strides)
            else:
                x = res_block(x, n_filters[i], strides=1, dilate=dilate)   # dilation added

    # model
    model = Model(inpt, x, name='backbone')

    return model


def res_block(x, n_filters, strides, se_ratio=0, dilate=1):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=1, activation='relu')
    x = Conv_BN(x, n_filters//4, 3, strides=strides, activation='relu', dilate=dilate)
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    if se_ratio:
        x = SE_block(x, se_ratio)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None, dilate=1):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False,
               dilation_rate=dilate)(x)
    x = FrozenBatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


def SE_block(inpt, ratio=16):     # spatial squeeze and channel excitation
    x = inpt
    n_filters = x._keras_shape[-1]
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,n_filters))(x)
    x = Dense(n_filters//ratio, activation='relu', use_bias=True)(x)
    x = Dense(n_filters, activation='sigmoid', use_bias=True)(x)
    x = multiply([inpt, x])
    return x


if __name__ == '__main__':

    model = resnet(input_shape=(224,224,3), depth=50)
    # model.summary()
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

