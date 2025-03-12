import torch
import collections, os, torch
import torch.nn as nn
from PIL import Image
import xmltodict
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
from torch_snippets import *
import glob
from ssd_utils.model import SSD300, MultiBoxLoss
from detect import *
from ssd_utils.utils import *
import pandas as pd
import numpy as np
import sklearn
import albumentations as A

DATA_ROOT = './chord_archive/'
IMAGE_ROOT = f'{DATA_ROOT}/images_train'
ANNOT_ROOT = f'./{DATA_ROOT}/annotations_train'
NOTE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'G#', 'F#', 'A#', 'fretboard']

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

annotations = sorted(glob.glob(ANNOT_ROOT+'/*'))
images = sorted(glob.glob(IMAGE_ROOT+'/*'))

augment_pipeline = A.Compose([
    # A.RandomCrop(width=400, height=400, p=0.3),
    A.Affine(scale=[0.5,2], translate_percent=[-0.05,0.05], shear=[-15,15], p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.Sharpen(p=0.3)
], bbox_params=A.BboxParams(format='pascal_voc', min_area=1000, min_visibility=0.1, label_fields=['class_labels']))

def transform(img, labels, boxes):
        transformed = augment_pipeline(image=img, bboxes=np.array(boxes), class_labels=np.array(labels))
        img = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['class_labels']

        # print(f'post augment boxes: {boxes}')
        # print(f'post augment labels: {labels}')

        return img, boxes, labels


class OpenDataset(torch.utils.data.Dataset):
    w, h = 300, 300
    def __init__(self, annotations, images):
        self.images = images
        self.annotations = annotations
        self.w = 300
        self.h = 300
    def __getitem__(self, ix):
        # load images and masks
        labels = []
        boxes = []
        img_path = self.images[ix] 
        img = np.array(Image.open(img_path))
        data_path = self.annotations[ix]
        with open(data_path, 'r') as file:
            data = xmltodict.parse(file.read())
        for obj in data['annotation']['object']:
            xMin = (float(obj['bndbox']['xmin']))
            yMin = (float(obj['bndbox']['ymin']))
            xMax = (float(obj['bndbox']['xmax']))
            yMax = (float(obj['bndbox']['ymax']))
            box = [xMin, yMin, xMax, yMax]
            if obj['category'] == 'finger':
                labels.append(obj['note'])
                boxes.append(box)
            if obj['category'] == 'fretboard':
                labels.append('fretboard')
                boxes.append(box)

        # print(f'pre augment boxes: {boxes}')
        # print(f'pre augment labels: {labels}')
    
        img, boxes, labels = transform(img, labels, boxes)
        if len(labels) == 0:
            return

        img = Image.fromarray(img).convert("RGB")
        img_size = list(img.size)

        scale = [self.w / img_size[0], self.h / img_size[1],self.w / img_size[0], self.h / img_size[1]]

        if boxes.ndim > 1:
            boxes[:,[0,2]] *= self.w / img_size[0] # normalize and scale
            boxes[:,[1,3]] *= self.h / img_size[1] # normalize and scale
        else: boxes *= scale
        
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255. #resize image and normalize
        return img, boxes, labels

    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for item in batch:
            if not item:
                continue
            img, image_boxes, image_labels = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device)/300.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
        images = torch.cat(images).to(device)
        # fretboard_boxes = torch.stack(fretboard_boxes)
        return images, boxes, labels
    def __len__(self):
        return len(self.images)

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
    # _fretboardbb = fretboard_model(images)

    ssd_loss = criterion(_regr, _clss, boxes, labels)
    # fretboard_loss = fretboard_criterion(_fretboardbb, fretboardbb)
    optimizer.zero_grad()
    # fretboard_optimizer.zero_grad()
    ssd_loss.backward()
    # fretboard_loss.backward()
    optimizer.step()
    # fretboard_optimizer.step()
    # return ssd_loss + fretboard_loss
    return ssd_loss
    
@torch.no_grad()
def validate_batch(inputs, model, criterion):
    model.eval()
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    return loss

n_epochs = 15

model = SSD300(num_classes, device)
# fretboard_model = FretboardModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# fretboard_optimizer = torch.optim.Adam(fretboard_model.parameters(), lr=1e-4)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)
# fretboard_criterion = nn.SmoothL1Loss()

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
    # print('pre augment:', original_image)
    original_image = transform(image=np.array(original_image))
    img = original_image['image']
    # print('post augment:', original_image)
    original_image = Image.fromarray(img).convert("RGB")
    bbs, labels, scores = detect(original_image, model, min_score=0.45, max_overlap=0.2,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]
    print(bbs, label_with_conf)
    
    show(original_image, bbs=bbs, texts=label_with_conf, text_sz=10)