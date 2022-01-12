import torch
from detr import detr
import numpy as np


# torch weights
# model = torch.load("weights/detr-r50.pth", map_location='cpu')['model']   # OrderedDict
model = torch.load("weights/detr-r50-dc5.pth", map_location='cpu')['model']   # OrderedDict
print(len(model.keys()))

# backbone:
back_weights = {k:v for k,v in model.items() if 'backbone' in k}   # conv,bn
print(len(back_weights))

# input_proj
input_proj_weights = {k:v for k,v in model.items() if 'input_proj' in k}  # conv
print(len(input_proj_weights))

# query_embed
query_emb = {k:v for k,v in model.items() if 'query_embed' in k}   # emb
print(len(query_emb))

# transformer encoder:
encoder_weights = {k:v for k,v in model.items() if 'encoder' in k}   # conv,ln,dense
print(len(encoder_weights))

# transformer decoder
decoder_weights = {k:v for k,v in model.items() if 'decoder' in k}   # conv,ln,dense
print(len(decoder_weights))

# mlp head: class_embed & bbox_embed
head_weights = {k:v for k,v in model.items() if 'class_' in k or 'bbox_' in k}   # dense
print(len(head_weights))

print('-------------------------------------run checking----------------------------------------')
# total_params = 0
# for k,v in back_weights.items():
#     print(k, v.shape)
#     tmp = 1
#     for j in list(v.shape):
#         tmp *= j
#     total_params += tmp
# print('total params', total_params)    # r50: 23561152


# total_params = 0
# for k,v in input_proj_weights.items():
#     print(k, v.shape)
#     tmp = 1
#     for j in list(v.shape):
#         tmp *= j
#     total_params += tmp
# print('total params', total_params)    # 2048->256: 524544


# total_params = 0
# for k,v in query_emb.items():
#     print(k, v.shape)
#     tmp = 1
#     for j in list(v.shape):
#         tmp *= j
#     total_params += tmp
# print('total params', total_params)    # emb: 25600


# total_params = 0
# for k,v in head_weights.items():
#     print(k, v.shape)
#     tmp = 1
#     for j in list(v.shape):
#         tmp *= j
#     total_params += tmp
# print('total params', total_params)    # dense weight & bias: 156256


# total_params = 0
# for k,v in encoder_weights.items():
#     if 'layers.0' not in k:
#         continue
#     print(k, v.shape)
#     tmp = 1
#     for j in list(v.shape):
#         tmp *= j
#     total_params += tmp
# print('total params', total_params)    # single encoder block: 1315072


# total_params = 0
# for k,v in decoder_weights.items():
#     if 'layers.0' not in k:
#         continue
#     print(k, v.shape)
#     tmp = 1
#     for j in list(v.shape):
#         tmp *= j
#     total_params += tmp
# print('total params', total_params)    # single decoder block: 1578752


print('-------------------------------------run converting----------------------------------------')
# keras model
enc_layers = 6
dec_layers = 6
keras_model = detr(input_shape=(512,512,3), pe='sine', n_classes=92,
                   depth=50, dilation=[False,False,False],
                   emb_dim=256, enc_layers=enc_layers, dec_layers=dec_layers, max_boxes=100)
# transpose conv: [out,in,k,k] -> [k,k,in,out]
# bn: [] -> []
for layer in keras_model.layers:
    if not layer.weights:
        continue
    print(layer.name)

    if layer.name == 'backbone':
        torch_weights = []
        layer_weights = []
        layer_name = 'conv1'
        for k,v in back_weights.items():
            # print(k, v.shape)
            v = v.numpy()
            if 'conv' in k or 'downsample.0' in k:
                v = np.transpose(v,(2,3,1,0))
            elif 'bn' in k:
                pass
            name = k.split('.')[-3]+k.split('.')[-2] if 'downsample' in k else k.split('.')[-2]
            if layer_weights and layer_name!=name:   # new layer
                torch_weights.append(layer_weights)
                layer_weights = []
                layer_name = name
            layer_weights.append(v)
        if layer_weights:
            torch_weights.append(layer_weights)

        idx = 0
        offset = [2,-1,1,-2]
        cnt = 0
        indices = [6,7,8,9, 26,27,28,29, 52,53,54,55, 90,91,92,93]   # r50 conv-id-path
        for sub_l in layer.layers:
            if sub_l.weights:
                print(layer.name, ': ', sub_l.name, idx, offset[cnt])
                # print(len(torch_weights[idx]))
                # print(len(sub_l.weights))
                if idx not in indices:
                    sub_l.set_weights(torch_weights[idx])
                else:
                    # conv1-conv2-bn1-bn2 vs conv2-bn2-conv1-bn1 situation
                    sub_l.set_weights(torch_weights[idx+offset[cnt]])
                    cnt = cnt+1 if cnt<3 else 0
                idx += 1

    if layer.name == 'input_proj':   # conv-bias
        torch_weights = [np.transpose(v,(2,3,1,0)) if 'weight' in k else v for k,v in input_proj_weights.items()]
        layer.set_weights(torch_weights)

    elif layer.name == 'query_embed':
        torch_weights = [v for k,v in query_emb.items()]
        layer.set_weights(torch_weights)

    elif 'encoder' in layer.name:
        # torch weights start from 0
        # keras model start from 1
        idx = int(layer.name.split('_')[-1]) - 1
        encoder_layer_weights = {k:v for k,v in encoder_weights.items() if 'layers.%d'%idx in k}
        msa_weights = [np.transpose(v, (1,0)) if 'weight' in k else v for k,v in encoder_layer_weights.items() if 'attn' in k]
        norm_weights = [v for k,v in encoder_layer_weights.items() if 'norm' in k]
        ffn_weights = [np.transpose(v, (1,0)) if 'weight' in k else v for k,v in encoder_layer_weights.items() if 'linear' in k]

        for sub_l in layer.layers:
            if sub_l.weights:
                print(layer.name, ': ', sub_l.name)
                if 'norm' in sub_l.name:   # weight & bias
                    sub_l.set_weights(norm_weights[0:2])
                    if norm_weights:
                        norm_weights = norm_weights[2:]
                elif 'att' in sub_l.name:
                    sub_l.set_weights(msa_weights)
                elif 'dense' in sub_l.name:    # weight & bias
                    sub_l.set_weights(ffn_weights[0:2])
                    if ffn_weights:
                        ffn_weights = ffn_weights[2:]

    elif 'decoder' in layer.name:
        idx = int(layer.name.split('_')[-1]) - 1
        decoder_layer_weights = {k:v for k,v in decoder_weights.items() if 'layers.%d'%idx in k}
        # print([[k, v.shape] for k,v in decoder_weights.items() if 'layers.%d' % idx in k])
        # print([i.shape for i in layer.get_weights()])
        self_weights = [np.transpose(v, (1,0)) if 'weight' in k else v for k,v in decoder_layer_weights.items() if 'self' in k]
        mutual_weights = [np.transpose(v, (1,0)) if 'weight' in k else v for k,v in decoder_layer_weights.items() if 'multi' in k]
        norm_weights = [v for k,v in decoder_layer_weights.items() if 'norm' in k]
        ffn_weights = [np.transpose(v, (1,0)) if 'weight' in k else v for k,v in decoder_layer_weights.items() if 'linear' in k]

        for sub_l in layer.layers:
            if sub_l.weights:
                print(layer.name, ': ', sub_l.name)
                if 'norm' in sub_l.name:   # weight & bias
                    sub_l.set_weights(norm_weights[0:2])
                    if norm_weights:
                        norm_weights = norm_weights[2:]
                elif 'self' in sub_l.name:
                    sub_l.set_weights(self_weights)
                elif 'mutual' in sub_l.name:
                    sub_l.set_weights(mutual_weights)
                elif 'dense' in sub_l.name:    # weight & bias
                    sub_l.set_weights(ffn_weights[0:2])
                    if ffn_weights:
                        ffn_weights = ffn_weights[2:]

    elif 'layer_normalization' in layer.name:   # last layer_norm
        layer.set_weights([decoder_weights['transformer.decoder.norm.weight'], decoder_weights['transformer.decoder.norm.bias']])

    elif 'cls_' in layer.name:
        torch_weights = [np.transpose(v, (1,0)) if 'weight' in k else v for k,v in head_weights.items() if 'class_' in k]
        layer.set_weights(torch_weights)

    elif 'box_' in layer.name:
        if 'pred' in layer.name:
            layer_weights = {k:v for k,v in head_weights.items() if 'bbox_embed.layers.2' in k}
        else:
            idx = int(layer.name.split('_')[-1]) - 1
            layer_weights = {k:v for k,v in head_weights.items() if 'bbox_embed.layers.%d'%idx in k}
        torch_weights = [np.transpose(v, (1,0)) if 'weight' in k else v for k,v in layer_weights.items()]
        layer.set_weights(torch_weights)

keras_model.save_weights('detr-r50-dc5.h5')




