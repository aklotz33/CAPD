from pathlib import Path
import cv2
import numpy as np
from subprocess import check_output

repo_dir = Path(check_output('git rev-parse --show-toplevel', shell=True, universal_newlines=True).strip())
scale_factor = 0.2

# load one image
imgs = list(map(str, Path(repo_dir/'images').glob('*.jpg')))
im = cv2.imread(imgs[0])

""" cv2.imshow('image',im)
cv2.waitKey(0) """

im_small = cv2.resize(im, (0,0), fx=0.2, fy=0.2) 
im_rows, im_cols, im_channels = im_small.shape

""" cv2.imshow('image small', im_small)
cv2.waitKey(0) """

cv_img = np.zeros(shape=(len(im), im_rows, im_cols, im_channels))
bad_logos = list()
file_names = list()
for i in range(len(imgs)):
    n= cv2.imread(imgs[i])
    file_names.append(imgs[i])
    if n is None:
        print('Got a bad logo')
        bad_logos.append(imgs[i])
        n = np.zeros(shape=(im_rows,im_cols,im_channels))
    else:
        n = cv2.resize(n, (im_rows, im_cols)) 

    cv_img[i,:,:,:] = n

(repo_dir/'data').mkdir(exist_ok=True)
np.save(repo_dir/'data'/'image_array.npy', cv_img)

with (repo_dir/'data'/'img_file_names.txt').open('w') as f:
    for item in file_names:
        f.write("%s\n" % item)

(repo_dir/'logs').mkdir(exist_ok=True)
with (repo_dir/'logs'/'bad_logos.txt').open('w') as f:
    for item in bad_logos:
        f.write("%s\n" % item)

print('done')