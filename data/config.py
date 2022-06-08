# config.py
from math import ceil

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    # 'min_sizes': [[75], [350], [1200]],  # TODO: 调整anchor大小
    'min_sizes': [
        [(ceil(340/16), ceil(200/16)), (ceil(300/16), ceil(250/16)), (ceil(280/16), ceil(200/16))],
        [(ceil(340/4), ceil(200/4)), (ceil(300/4), ceil(250/4)), (ceil(280/4), ceil(200/4))],
        [(340, 200), (300, 250), (280, 200)]
    ],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 40,
    'decay1': 20,
    'decay2': 30,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
