from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import glob

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/2021-11-30/Resnet50_best.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.8, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=10, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=10, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.8, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu=False):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def plot(img, boxes, landmarks):
    """
    plot pic in pic.

    Args:
        img ([type]): [description]
        boxes ([type]): [description]
        landmarks ([type]): [description]
    """
    for box, landmark in zip(boxes, landmarks):
        x1, y1, x2, y2 = list(map(lambda x: int(round(x)), box))
        
        # crop patch on raw img and plot boundary
        patch = img[y1: y2, x1: x2].copy()
        cv2.rectangle(patch, (0, 0), (x2 - x1, y2 - y1), (0, 255, 255), 3)
        
        # calculate the relative coordinates of keypoints to the box which it belongs
        landmark[::2] -= x1
        landmark[1::2] -= y1

        # plot keypoints on patch
        for i in range(0, len(landmark), 2):
            cv2.circle(patch, (landmark[i], landmark[i + 1]), 10, (0, 0, 255), -1)
        
        # scale patch by ratio 0.5
        h_new = int(round((y2 - y1) * 0.5))
        w_new = int(round((x2 - x1) * 0.5))
        patch = cv2.resize(patch, (w_new, h_new))
        
        # plot patch on raw img
        img[-h_new:, -w_new:] = patch


if __name__ == '__main__':
    save_dir = "./data/model_out"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    # net = load_model(net, args.trained_model, args.cpu)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # print(net)
    # cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    img_dir = "./data/model_testimg/"
    img_paths = glob.glob(img_dir + "*.jpg")
    # save_preds_dir = "/media/yee/WorkSpace/mAP-master/input/detection-results/"
    for i, img_path in enumerate(img_paths):
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        landm_file = img_path.replace("jpg", "txt")

        img = np.float32(img_raw)

        h_raw, w_raw, _ = img.shape

        img_dim = cfg_re50["image_size"]
        ratio = img_dim / h_raw if h_raw > w_raw else img_dim / w_raw
        h_new = int(round(h_raw * ratio))
        w_new = int(round(w_raw * ratio))
        img = cv2.resize(img, (w_new, h_new))

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(h_new, w_new))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors
        boxes = decode(loc.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / ratio
        boxes = boxes.detach().cpu().numpy()
        scores = conf.squeeze(0).detach().cpu().numpy()[:, 1]
        landms = decode_landm(landms.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / ratio
        landms = landms.detach().cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_raw - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_raw - 1)

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis]))
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # save landms
        if len(dets) == 0:
            print("no dets")
            continue
        b = dets[0]
        landms = "{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format(
            b[5], b[6], b[7], b[8], b[9], b[10]
        )
        with open(landm_file, "w") as f:
            f.write(landms)

        # show image
        # lines = []
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(lambda x: int(round(x)), b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cx, cy = b[0], b[1] + 25
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 5, (0, 0, 255), -1)
                cv2.circle(img_raw, (b[7], b[8]), 5, (0, 127, 255), -1)
                cv2.circle(img_raw, (b[9], b[10]), 5, (255, 0, 255), -1)

                # # save landms
                # landms = "{:2f}, {:2f}, {:2f}, {:2f}, {:2f}, {:2f}\n".format(
                #     b[5], b[6], b[7], b[8], b[9], b[10]
                # )
                # with open(landm_file, "w") as f:
                #     f.write(landms)
                
                # lines.append("rect {} {} {} {} {} {} {} {} {} {} {}\n".format(
                #     text, b[0], b[1], b[2], b[3], b[5], b[6], b[7], b[8], b[9], b[10]))
            
            # plot(img_raw, dets[:, :4], dets[:, 5:])

            # save image
            name = os.path.join(save_dir, "res_{}".format(os.path.basename(img_path)))
            cv2.imwrite(name, img_raw)

            # # save preds
            # save_path = os.path.join(save_preds_dir, os.path.basename(img_path).replace("jpg", "txt"))
            # with open(save_path, "w") as f:
            #     f.writelines(lines)
