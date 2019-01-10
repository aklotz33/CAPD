from pathlib import Path
import cv2

from src.util import REPO_PATH


img_path = REPO_PATH / 'images'
downscaled_path = REPO_PATH / 'downscaled_images'

downscaled_path.mkdir(exist_ok=True)
for img_file in list(Path(img_path).glob('*.jpg')):
    img = cv2.imread(str(img_file))
    img = cv2.resize(img,(180,180))
    cv2.imwrite(str(downscaled_path / img_file.stem) + '_small.jpg', img)