

import glob
import cv2
import numpy as np

img_path = "images/*.jpg"
scale_factor = 0.2

# load one image
imgs = glob.glob(img_path)
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


np.save('image_array', cv_img)

with open('bad_logos.txt', 'w') as f:
    for item in bad_logos:
        f.write("%s\n" % item)


with open('your_file.txt', 'w') as f:
    for item in file_names:
        f.write("%s\n" % item)

print('done')