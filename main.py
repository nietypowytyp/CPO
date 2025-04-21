import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fastai.vision.all import *
from PIL import Image
from fastai.vision.utils import show_image

torch.cuda.empty_cache()

def clean_images(imgs):
    for img_path in imgs:
        try:
            img = PILImage.create(img_path)
            t = ToTensor()(img)
            if not torch.isfinite(t).all():
                print(f"Bad tensor: {img_path}")
                img_path.unlink()
        except Exception as e:
            print(f"Corrupt image: {img_path}, Error: {e}")
            img_path.unlink()

if __name__ == '__main__':
    path = Path('hand_gestures/images')

    print("Image count:", len(get_image_files(path)))

    searches = [folder.name for folder in path.iterdir() if folder.is_dir()]

    imgs = get_image_files(path)
    print(f"Checking {len(imgs)} images...")
    clean_images(imgs)

    print("Final image count:", len(get_image_files(path)))

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='pad', pad_mode='zeros')]
    ).dataloaders(path, bs=32)

    print("Classes:", dls.vocab)
    print("Number of classes:", dls.c)
    dls.show_batch(max_n=6)
    plt.show()

    img = PILImage.create('hand_gestures/images/fingerSymbols/IMG_20220430_181702.jpg')
    print(img.mode)


    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.freeze()
    learn.fit_one_cycle(1, 1e-3)  # Safe test run

    # learn = vision_learner(dls, resnet18, metrics=error_rate)
    # learn.lr_find()
    # learn.fine_tune(3, base_lr=1e-3)

    hand_gesture,_,probs = learn.predict(PILImage.create('hand_gestures/images/fingerSymbols/IMG_20220430_181702.jpg'))
    print(f"This is a: {hand_gesture}.")
    print(f"Probability: {probs[0]:.4f}")

    img_path = 'hand_gestures/images/fingerSymbols/IMG_20220430_181702.jpg'
    img = mpimg.imread(img_path)

    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title('Finger Symbol')
    plt.show()