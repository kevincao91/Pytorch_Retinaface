"""
    @author: yee
    @date: 2021/1/20
    @description: 
"""
import cv2
import glob
from tqdm import tqdm


img_dir = "./data/custom/train/images/"
img_paths = glob.glob(img_dir + "*jpg")

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
size = (2560, 1440)
fps = 3
writer = cv2.VideoWriter()
path = "test.avi"
writer.open(path, fourcc, fps, size, True)

for img_path in tqdm(img_paths[:60]):
    img = cv2.imread(img_path)
    writer.write(img)

writer.release()
print("done")
