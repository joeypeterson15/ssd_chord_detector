import torch
from ssd_utils.model import SSD300
from detect import *
import glob
import albumentations as A
import numpy as np

NOTE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'G#', 'F#', 'A#', 'fretboard']

label2target = {l:t+1 for t,l in enumerate(NOTE_LABELS)}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

IMAGE_ROOT = './chord_archive/images_train'
n_classes, device = 12, 'cpu'
model = SSD300(n_classes, device)
model.load_state_dict(torch.load('./ssd_chord_model.pth', weights_only=True))
model.eval()

images = sorted(glob.glob(IMAGE_ROOT+'/*'))


transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    A.Affine(scale=[0.25,2], translate_percent=[-0.15,0.15], shear=[-5,5], p=0.3),
    A.RandomBrightnessContrast(p=0.4),
], seed=137, strict=True)

for n in range(10, 20):
    img_path = images[n]
    print('\n')
    print('img path: ', img_path)
    original_image = Image.open(img_path, mode='r')
    original_image = transform(image=np.array(original_image))
    img = original_image['image']
    original_image = Image.fromarray(img).convert("RGB")
    bbs, labels, scores = detect(original_image, model, min_score=0.45, max_overlap=0.2,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
    print(bbs, label_with_conf)
    
    show(original_image, bbs=bbs, texts=label_with_conf, text_sz=10)