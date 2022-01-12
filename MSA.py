from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.activations import relu
import tensorflow as tf
import keras.backend as K
import math


def gelu(x, approx=False):
    if approx:
        return 0.5 * x * (1 + K.tanh(K.sqrt(K.constant(2./math.pi)) * (x + 0.044715 * K.pow(x, 3))))
    else:
        return 0.5 * x * (1. + tf.math.erf(x / K.sqrt(K.constant(2.))))


# MSA layer
class MultiHeadAttention(Model):
    def __init__(self, model_size, num_heads, attn_drop=0., ffn_drop=0., **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads
        self.WQKV = Dense(3*model_size)
        self.dense = Dense(model_size)
        self.msa_drop = Dropout(attn_drop)
        self.mlp_drop = Dropout(ffn_drop)

    def call(self, inputs, mask=None):
        # query: (batch, maxlen, model_size)
        # key  : (batch, maxlen, model_size)
        # value: (batch, maxlen, model_size)
        batch_size = tf.shape(inputs[0])[0]

        # shape: (batch, maxlen, model_size)
        qkv = self.WQKV(tf.concat(inputs, axis=1))
        splits = [K.int_shape(i)[1] for i in inputs]
        query, key, value = tf.split(qkv, splits, axis=1)

        def _split_heads(x):
            seq_len = x.shape[1]
            x = tf.reshape(x, shape=[batch_size, seq_len, self.num_heads, self.head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])

        # shape: (batch, num_heads, maxlen, head_size)
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        # shape: (batch, num_heads, maxlen, maxlen)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # 缩放 matmul_qk
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)

        if isinstance(mask, list):
            mask = mask[0]
        if mask is not None:
            score += (1 - mask) * -1e9     # add mask=0 points with -inf, results in 0 in softmax

        # softmax & dropout
        alpha = tf.nn.softmax(score)
        alpha = self.msa_drop(alpha)

        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        query_len = context.shape[1]
        context = tf.reshape(context, (batch_size, query_len, self.model_size))
        output = self.dense(context)
        output = self.mlp_drop(output)

        return output

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[0]
        return (B,N,self.model_size)


# FFN layer
class FeedForwardNetwork(Model):
    def __init__(self, dff_size, model_size, activation=relu, drop_rate=0.):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = Dense(dff_size, activation=activation)   # relu/gelu
        self.dense2 = Dense(model_size)
        if drop_rate:
            self.drop1 = Dropout(drop_rate)
            self.drop2 = Dropout(drop_rate)
        self.act = Activation(activation)
        self.model_size = model_size
        self.drop_rate = drop_rate

    def call(self, x):
        x = self.dense1(x)
        x = self.act(x)
        if self.drop_rate:
            x = self.drop1(x)
        x = self.dense2(x)
        if self.drop_rate:
            x = self.drop2(x)
        return x

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape
        return (B,N,self.model_size)


if __name__ == '__main__':

    # test MSA & FFN layer
    x = Input((20, 10))   # query, [N1,D]
    x1 = Input((30, 10))  # key, [N2,D]
    mask = Input((20,30))  # [N_q, N_k]
    y = MultiHeadAttention(10, 2)(inputs=[x,x1,x1], mask=mask)
    print('joint', y)
    y = MultiHeadAttention(10, 2)(inputs=[y,y,y], mask=None)
    print('self', y)
    y = FeedForwardNetwork(16, 10)(y)

    model = Model([x,x1,mask],y)
    # model.summary()

    layer = MultiHeadAttention(10, 2)
    y = layer([y,y,y])
    print(layer.weights)
    for l in layer.layers:
        print(l.name)
        print(l.weights)

