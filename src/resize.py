import os
import cv2

from .util import REPO_STR


img_path = REPO_STR + '/images'
downscaled_path = REPO_STR + '/downscaled_images'

if not os.path.exists(downscaled_path):
    os.mkdir(downscaled_path)

for img_file in os.listdir(img_path):
    img = cv2.imread(img_path+img_file)
    img = cv2.resize(img,(180,180))
    cv2.imwrite(downscaled_path+img_file, img)