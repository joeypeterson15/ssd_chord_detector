import torch
import collections, os, torch
from PIL import Image
import xmltodict
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
from torch_snippets import *
import glob
from ssd_utils.model import SSD300, MultiBoxLoss
from ssd_utils.detect import *
from ssd_utils.utils import *
import pandas as pd
import numpy as np
import sklearn

DATA_ROOT = './chord_archive/'
IMAGE_ROOT = f'{DATA_ROOT}/images_train'
ANNOT_ROOT = f'./{DATA_ROOT}/annotations_train'
# NOTE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'G#', 'F#', 'A#']
NOTE_LABELS = ['Note']

label2target = {l:t+1 for t,l in enumerate(NOTE_LABELS)}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()

class OpenDataset(torch.utils.data.Dataset):
    w, h = 300, 300
    def __init__(self, annotations, images):
        # self.image_dir = image_dir
        self.images = images
        # self.annot_root = annot_dir
        self.annotations = annotations
        self.w = 300
        self.h = 300
        # logger.info(f'{len(self)} items loaded')
    def __getitem__(self, ix):
        # load images and masks
        img_path = self.images[ix] 
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.
        data_path = self.annotations[ix]
        with open(data_path, 'r') as file:
            data = xmltodict.parse(file.read())
        labels = []
        boxes = []
        original_width = int(data['annotation']['size']['width'])
        original_height = int(data['annotation']['size']['height'])
        for obj in data['annotation']['object']:
            if obj['category'] == 'finger':
                xMin = (float(obj['bndbox']['xmin']) / original_width) * self.w
                yMin = (float(obj['bndbox']['ymin']) / original_height) * self.h
                xMax = (float(obj['bndbox']['xmax']) / original_width) * self.w
                yMax = (float(obj['bndbox']['ymax']) / original_height) * self.h
                box = [xMin, yMin, xMax, yMax]
                # box = box.astype(np.uint32).tolist()
                boxes.append(box)
                labels.append('Note')

        return img, boxes, labels

    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for item in batch:
            img, image_boxes, image_labels = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device)/300.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
        images = torch.cat(images).to(device)
        return images, boxes, labels
    def __len__(self):
        return len(self.images)

annotations = sorted(glob.glob(ANNOT_ROOT+'/*'))
images = sorted(glob.glob(IMAGE_ROOT+'/*'))
assert len(images) == len(annotations)
TRAIN_PERCENT = .95
split = int(len(images) * TRAIN_PERCENT)

train_images = images[:split]
val_images = images[split:]

train_annots = annotations[:split]
val_annots = annotations[split:]

train_ds = OpenDataset(train_annots, train_images)
test_ds = OpenDataset(val_annots, val_images)

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn, drop_last=True)


def train_batch(inputs, model, criterion, optimizer):
    model.train()
    N = len(train_loader)
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    
@torch.no_grad()
def validate_batch(inputs, model, criterion):
    model.eval()
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    return loss


n_epochs = 10

model = SSD300(num_classes, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

for epoch in range(n_epochs):
    LOSS = 0
    print(f'epoch: {epoch + 1}')
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss = train_batch(inputs, model, criterion, optimizer)
        LOSS += loss.item()

    print(f'loss: {LOSS}')
    # _n = len(test_loader)
    # for ix,inputs in enumerate(test_loader):
    #     loss = validate_batch(inputs, model, criterion)

for n in range(20):
    img_path = images[n]
    original_image = Image.open(img_path, mode='r')
    # bbs, labels, scores = detect(original_image, model, min_score=0.9, max_overlap=0.5,top_k=200, device=device)
    bbs, labels, scores = detect(original_image, model, min_score=0.4, max_overlap=0.2,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
    print(bbs, label_with_conf)
    show(original_image, bbs=bbs, texts=label_with_conf, text_sz=10)