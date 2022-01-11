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
    frozenBN: 就是frozen的BN，weight/bias/mean/var都是load_pretrained，不可训练也不可修改
    dilation: torchvision自带的resnet, 有一个replace_stride_with_dilation参数，必须是3-element tuple，用来控制stage3/4/5的空洞卷积
       * 加在指定stage的第二个及之后的bottleneck block的3x3卷积层里面
       * dilation=stride(=2)，然后取消改stage的下采样(stride=1)

    ----- PE -------
    PE-sine: 2d map上的PE，由x轴&y轴两个维度的PE组成
    PE-learned: 也是两个轴的embedding concat在一起组成
    这个pe不是加在transformer的输入上，而是加在了每个transformmer block的输入上,
    这个pe是专门给input feature map用的，decoder input的pe是learnable的

    ----- transformer ------
    pre_norm: 默认是在输入时加layer norm
    qkv: encoder的qkv都来自input feature，decoder的qkv有两种，第一层自注意力qkv来自targets，第二层交互注意力，q来自targets，kv来自inputs

    ------ head ------
    individual cls & box heads，dense
    tricky n_classes: COCO的label idx是[1,90]，所以给了no object的id是91，然后预测向量的维度是91+1，id0是dummy label


## deformable DETR

    paper: DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION, 2021, SenseTime
    official: https:// github.com/fundamentalvision/Deformable-DETR





