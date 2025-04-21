import os
from fastai.vision.all import *
from PIL import Image

# im = Image.open("hand_gestures/images/closedFist/IMG_20220430_180531.jpg")
# im.thumbnail((250, 250))
# im.show()

if __name__ == '__main__':
    path = Path('hand_gestures/images')

    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(len(failed))