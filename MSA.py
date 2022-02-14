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
    def __init__(self, model_size, num_heads, attn_drop=0.1, ffn_drop=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads
        self.WQ = Dense(model_size, name='dense_q')   # [b,Nq,d]
        self.WK = Dense(model_size, name='dense_k')   # [b,Nk,d]
        self.WV = Dense(model_size, name='dense_v')   # [b,Nv,d], Nk=Nv
        self.dense = Dense(model_size)
        self.msa_drop = Dropout(attn_drop)
        self.mlp_drop = Dropout(ffn_drop)

    def call(self, inputs, mask=None):
        # query: (batch, maxlen, model_size)
        # key  : (batch, maxlen, model_size)
        # value: (batch, maxlen, model_size)

        query, key, value = inputs
        batch_size = tf.shape(query)[0]

        # in_proj: shape: (batch, maxlen, model_size)
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, x.shape[1], self.num_heads, self.head_size])
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
        alpha = tf.nn.softmax(score)    # [b,Nq,Nk]
        alpha = self.msa_drop(alpha)

        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, query.shape[2], self.model_size))
        output = self.dense(context)
        output = self.mlp_drop(output)

        return output

    def compute_output_shape(self, input_shape):
        B, N, _ = input_shape[0]
        return (B,N,self.model_size)


# FFN layer: dense-relu-drop-dense-drop
class FeedForwardNetwork(Model):
    def __init__(self, emb_dim=256, hidden_dim=2048, activation=relu, drop_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = Dense(hidden_dim, activation=activation)   # relu/gelu
        self.dense2 = Dense(emb_dim)
        if drop_rate:
            self.drop1 = Dropout(drop_rate)
            self.drop2 = Dropout(drop_rate)
        self.act = Activation(activation)
        self.emb_dim = emb_dim
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
        return (B,N,self.emb_dim)


if __name__ == '__main__':

    # test MSA & FFN layer
    q = Input((20,10))  #  [b,N1,d]
    k = Input((40,10))   # [b,N2,d]
    mask = Input((20,40))     # [Nq,Nk]
    y = MultiHeadAttention(10, 2)(inputs=[q,k,k], mask=mask)
    y = MultiHeadAttention(10, 2)(inputs=[y,k,k], mask=mask)
    y = FeedForwardNetwork(16, 10)(y)

    model = Model([q,k,mask],y)
    # model.summary()

    print(model.weights)

