from models.retinaface import RetinaFace
from data import cfg_re50
import netron
import torch


sample = torch.rand(1, 3, 860, 860)
model = RetinaFace(cfg_re50)
output = model(sample)
 
onnx_path = "retinaface.onnx"
torch.onnx.export(model, output, onnx_path)
 
netron.start(onnx_path)
