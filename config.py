"""
    Author: yee
    Date: 2021/1/18
    Description:
"""
from math import ceil


class ResnetConfig(object):
    name = "Resnet50"
    model_path = "./weights/2021-1-14/Resnet50_best.pth"
    steps = [8, 16, 32]
    # anchor_size = [(340, 200), (300, 250), (280, 200)]
    min_sizes = [
        [(ceil(dim[0] / down), ceil(dim[1] / down)) for dim in [(340, 200), (300, 250), (280, 200)]]
        for down in (16, 4, 1)
    ]
    variance = [0.1, 0.2]
    clip = False
    loc_weight = 2
    epoch = 30
    batch_size = 4
    use_gpu: True
    ngpu = 1
    decay = [15, 20]
    img_size = 640
    pretrain = True
    return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    in_channel = 256
    out_channel = 256
    conf_thresh = 0.8
    nms_thresh = 0.2
    top_k = 10


res_cfg = vars(ResnetConfig)
