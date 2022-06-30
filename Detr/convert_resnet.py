import torch
from resnet import resnet
import numpy as np


model = torch.load("weights/detr-r50.pth", map_location='cpu')['model']   # OrderedDict
print(len(model.keys()))

# backbone:
back_weights = {k:v for k,v in model.items() if 'backbone' in k}   # conv,bn
print(len(back_weights))


torch_weights = []
layer_weights = []
layer_name = 'conv1'
for k,v in back_weights.items():
    print(k, v.shape)
    v = v.numpy()
    if 'conv' in k or 'downsample.0' in k:
        v = np.transpose(v,(2,3,1,0))
    elif 'bn' in k:
        pass     # torch:[weight,bias,mean,var]  vs  keras:[gamma,beta,mean,var]
    name = k.split('.')[-3]+k.split('.')[-2] if 'downsample' in k else k.split('.')[-2]
    if layer_weights and layer_name!=name:   # new layer
        torch_weights.append(layer_weights)
        layer_weights = []
        layer_name = name
    layer_weights.append(v)
if layer_weights:
    torch_weights.append(layer_weights)

print(len(torch_weights))


# keras model
keras_model = resnet(input_shape=(800,1201,3), depth=50, dilation=False)
# keras_model.summary()


cnt = 0
for layer in keras_model.layers:
    if not layer.weights:
        continue
    print(layer.name)
    print(layer.weights)   # tf.variable
    print([i.shape for i in layer.get_weights()])

    offset = [2,-1,1,-2]
    indices = [6,7,8,9, 26,27,28,29, 52,53,54,55, 90,91,92,93]   # r50 conv-id-path

    # print(len(torch_weights[idx]))
    # print(len(sub_l.weights))
    if cnt not in indices:
        layer.set_weights(torch_weights[cnt])
    else:
        # keras:conv0-conv3-bn0-bn3 vs torch:conv3-bn3-conv0-bn0
        print('here')
        idx = indices.index(cnt) % 4
        print(idx, offset[idx])
        layer.set_weights(torch_weights[cnt+offset[idx]])

    cnt += 1

keras_model.save_weights('r50.h5')



