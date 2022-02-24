## DETR

    paper: End-to-End Object Detection with Transformers, 2020, facebook
    official repo: https://github.com/facebookresearch/detr

    特色就是fully end-to-end，是一个颠覆性的检测架构，用transformer encoder-decoder直接输出box predicitons

    两种模型
    DETR：origin ResNet back
    DETR-DC：修改了ResNet的最后一个阶段，引入distillation conv

    NestedTensor: 用来支持不同大小的图片
    * pad_img: 左上角开始填充图像
    * mask: 有图像的地方是0，没图像的地方是1
    * batch_shape: 是这个batch内图片的最大尺寸

    pretrained: 
    * detr-r50.pth
    * detr-r50-dc5.pth


    ----- backbone ------
    resnet back: 这里面涉及一个数据结构NestedTensor，每个样本自带一张pad mask，residual的stride2是放在1-3-1的3里面
    frozenBN: 就是freeze的BN，weight/bias/mean/var都是load_pretrained，不可训练也不可修改
    dilation: torchvision自带的resnet, 有一个replace_stride_with_dilation参数，必须是3-element tuple，用来控制stage3/4/5的空洞卷积
       * 加在指定stage的第二个及之后的bottleneck block的3x3卷积层里面
       * dilation=stride(=2)，然后取消改stage的下采样(stride=1)
    !!！还有一个发现，预训练权重的resnet的stem和conv1也是frozen的，torch model load进来也保持frozen
    load_weights时候需要注意模型和权重中层的trainable state要一致，否则会出现ValueError: axes don't match array

    ----- PE -------
    PE-sine: 2d map上的PE，由x轴&y轴两个维度的PE组成
    PE-learned: 也是两个轴的embedding concat在一起组成
    这个pe不是加在transformer的输入上，而是加在了每个transformmer block的输入上,
    这个pe是加给[QKV]中的Q和K
    encoder的PE是sine的(包括decoder中来自encoder的K)
    decoder的PE是learned的(只给到decoder mutual-att-block中的Q和decoder self-att-block中的QK)

    ----- transformer ------
    pre_norm: 默认是在输入时加layer norm
    qkv: encoder的qkv都来自input feature，decoder的qkv有两种，第一层自注意力qkv来自targets，第二层交互注意力，q来自targets，kv来自inputs

    ------ head ------
    individual cls & box heads，dense
    tricky n_classes: COCO的label idx是[1,90]，所以给了no object的id是91，然后预测向量的维度是91+1，id0是dummy label


    ------ training details ------
    AdamW
    separated learning rate: backbone 1e-5, transformer 1e-4
    weight decay = 1e-4
    init: backbone - ImageNet-pretrained with frozen BN, transformer - Xavier
    augmentation: scale, shortest in [480,800], longest side < 1333, random crop, prob=0.5 
    transformer dropout: 0.1
    300 epochs, lr decay by 10 after 200 epochs
    loss: linear combine l1 & giou


    ------ torch parameters() & buffers() ------
    这是源代码里涉及到的一个用法，detr的backbone里面有两种参数：
    * 一种是conv weights，trainable，可以被梯度更新，可以通过model.parameters() / model.named_parameters()看到
    * 一种是Frozen BN weights，[gamma，beta，weight，bias]，non-trainable，在源代码里直接设计成了variable，通过model.buffers() / model.named_buffers()来查看
    * 两种参数都会通过model.state_dict()被保存为OrderedDict


    ----- train one epoch -----
    scipy.optimize.linear_sum_assignment: 
    ref1: https://stackoverflow.com/questions/62238064/how-to-use-scipy-optimize-linear-sum-assignment-in-tensorflow-or-keras
    ref2: https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py
    ref3: https://github.com/Visual-Behavior/detr-tensorflow/blob/main/detr_tf/loss/hungarian_matching.py


    tf.scatter_nd_update:
    这个方法有个替身tf.tensor_scatter_nd，
    刚开始用scatter_nd_update一直报错，发现是ref和updates必须都是tf.Variable，不能是tf.Constant


    ----- 训练心得 -----
    0. 用一两张去预实验，首先验证了网络能够正确收敛到target上，然后用大数据集去训练：
    1. 不稳定，loss忽高忽低
    2. 收敛慢


## deformable DETR

    paper: DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION, 2021, SenseTime
    official: https:// github.com/fundamentalvision/Deformable-DETR







