import sys
from pathlib import Path
import cv2

from util import REPO_PATH


def downscale(size=180):
    """
    Reduces the size of all images in the images folder. Saves new image files in downscaled_images folder.
    :param size: (int) height and width of new images in pixels
    :return: None
    """
    img_path = REPO_PATH / 'images'
    downscaled_path = REPO_PATH / 'downscaled_images'

    downscaled_path.mkdir(exist_ok=True)
    for img_file in list(Path(img_path).glob('*.jpg')):
        img = cv2.imread(str(img_file))
        img = cv2.resize(img, (size, size))
        cv2.imwrite(str(downscaled_path / img_file.stem) + '_small.jpg', img)


if __name__ == '__main__':
    size = 180
    if len(sys.argv)>1:
        size = int(sys.argv[1])
    downscale(size)
