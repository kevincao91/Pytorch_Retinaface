"""
Author: yee
Date: 2021/1/18
Description:
"""
import torch
import cv2
import time
import numpy as np
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from config import res_cfg as cfg
from calCenter import trans_image


class AlignDetector(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def init_model(self):
        """
        initialize model.

        Returns:
            model: initialized Retinaface model
        """
        print("[INFO] Load model from {}".format(cfg["model_path"]))
        model = RetinaFace(cfg=cfg, phase="test")
        model.load_state_dict(torch.load(cfg["model_path"]))
        model.eval()
        model = model.to(self.device)
        self.model = model
        print("[INFO] Model loaded successful")
    
    def inference(self, img):
        """
        predict hopper and keypoints for input img.

        Args:
            img (numpy array): the img to be predict, shape(h, w, c)

        Returns:
            preds (numpy array): predict results, shape(n, 11->[boxes, confidence, keypoints])
        """
        # pre-process
        h_raw, w_raw = img.shape[:2]
        ratio = cfg["img_size"] / h_raw if h_raw > w_raw else cfg["img_size"] / w_raw
        h_new = int(round(h_raw * ratio))
        w_new = int(round(w_raw * ratio))
        img_resized = cv2.resize(img, (w_new, h_new))
        img_resized = img_resized.astype(np.float32)
        
        # ndarray -> tensor & shape to (1, c, h, w)
        img_resized = torch.from_numpy(img_resized)
        img_resized = img_resized.permute(2, 0, 1).unsqueeze(0)
        img_resized = img_resized.to(self.device)

        # forward pass
        tic = time.time()
        loc, conf, landms = self.model(img_resized)
        # print("[INFO] Model forward time: {:.4f}s".format(time.time() - tic))

        # boxes & landmarks decode
        priors = PriorBox(cfg, (h_new, w_new)).forward().to(self.device)
        boxes = decode(loc.squeeze(0), priors, cfg['variance'])
        landms = decode_landm(landms.squeeze(0), priors, cfg['variance'])

        # scale back to raw image size
        box_scale = torch.tensor([w_new, h_new] * 2).to(self.device)
        landm_scale = torch.tensor([w_new, h_new] * 3).to(self.device)
        boxes *= box_scale / ratio
        landms *= landm_scale / ratio

        # cuda tensor -> cpu ndarray
        boxes = boxes.detach().cpu().numpy()
        landms = landms.detach().cpu().numpy()
        scores = conf.squeeze(0).detach().cpu().numpy()[:, 1]

        # filter predictions which confidence less than the threshold
        inds = np.where(scores > cfg["conf_thresh"])[0]
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

        # keep topk before NMS
        inds = scores.argsort()[::-1][:cfg["top_k"]]
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

        # constrained within the boundary
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_raw - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_raw - 1)

        # do NMS
        preds = np.concatenate((boxes, scores[:, np.newaxis]), axis=1)
        keep = py_cpu_nms(preds, cfg["nms_thresh"])
        preds, landms = preds[keep], landms[keep]
        
        preds = np.concatenate((preds, landms), axis=1)
        return preds

    def detect(self, img, save=False):
        """
        detect hopper and keypoints for input img, and calculate the center of hopper.

        Args:
            img (numpy array): the img to be detect, shape(h, w, c)
            save       (bool): whether to save result img, default False

        Returns:
            
        """
        preds = self.inference(img)
        if len(preds) == 0:
            print("[INFO] No hopper detected")
            return
        
        boxes, landmarks = preds[:, :4], preds[:, 5:]
        
        # calculate center of the hopper
        centers = np.zeros((preds.shape[0], 2), dtype=np.float32)
        for i, landmark in enumerate(landmarks):
            cx, cy = trans_image(landmark.reshape(-1, 2))
            centers[i] = [cx, cy]
        
        self.plot(img, boxes, landmarks, centers, save)
        
        return centers
        
    @staticmethod
    def plot(img, boxes, landmarks, centers, save):
        """
        plot preds in origin img, form as pic-in-pic.

        Args:
            img       (numpy array): origin img
            boxes     (numpy array): coordinates of boxes, shape(n, 4)
            landmarks (numpy array): coordinates of keypoints, shape(n, 6)
            centers   (numpy array): coordinates of centers, shape(n, 2)
            save             (bool): whether to save img
        
        Returns:
            None
        """
        assert boxes.shape[0] == 1, "Currently only supports one prediction box"
        for box, landmark, center in zip(boxes, landmarks, centers):
            x1, y1, x2, y2 = list(map(lambda x: int(round(x)), box))
            
            # crop patch on raw img and plot boundary
            patch = img[y1: y2, x1: x2].copy()
            cv2.rectangle(patch, (0, 0), (x2 - x1, y2 - y1), (0, 255, 0), 3)
            
            # calculate the relative coordinates of keypoints and center
            landmark[::2] -= x1
            landmark[1::2] -= y1
            center[0] -= x1
            center[1] -= y1

            # plot keypoints on patch
            for i in range(0, len(landmark), 2):
                cv2.circle(patch, (landmark[i], landmark[i + 1]), 10, (255, 0, 255), -1)
            
            # plot center on patch
            cv2.circle(patch, (center[0], center[1]), 10, (0, 0, 255), -1)
            
            # scale patch by ratio 0.5
            h_new = int(round((y2 - y1) * 0.5))
            w_new = int(round((x2 - x1) * 0.5))
            patch = cv2.resize(patch, (w_new, h_new))
            
            # plot patch on raw img
            img[-h_new:, -w_new:] = patch

            if save:
                cv2.imwrite("res.jpg", img)


if __name__ == "__main__":
    detector = AlignDetector()
    detector.init_model()
    img_path = "./curve/140.jpg"
    img = cv2.imread(img_path)
    detector.detect(img, save=True)
