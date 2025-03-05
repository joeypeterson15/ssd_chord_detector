import albumentations as A
import numpy as np
from PIL import Image
import glob
import xmltodict

DATA_ROOT = './chord_archive/'
IMAGE_ROOT = f'{DATA_ROOT}/images_train'
ANNOT_ROOT = f'{DATA_ROOT}/annotations_train'

annotations = sorted(glob.glob(ANNOT_ROOT+'/*'))
images = sorted(glob.glob(IMAGE_ROOT+'/*'))

image_transform_pipeline = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomBrightnessContrast(p=1),
    A.Sharpen(p=1)
])

bbox_transform_pipeline = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomBrightnessContrast(p=1),
    A.Sharpen(p=1)
], bbox_params=A.BboxParams(format='coco'))

def augment_images(images_dir, annots_dir):
    for i in range(len(images)):
        pil_image = Image.open(images[i])
        image = np.array(pil_image)
        transformed_image = image_transform_pipeline(image)

        annot_data = annots_dir[i]
        with open(annot_data, 'r') as file:
            data = xmltodict.parse(file.read())

        original_width = int(data['annotation']['size']['width'])
        original_height = int(data['annotation']['size']['height'])
        for obj in data['annotation']['object']:
            if obj['category'] == 'finger':
                xMin = (float(obj['bndbox']['xmin']) / original_width)
                yMin = (float(obj['bndbox']['ymin']) / original_height)
                xMax = (float(obj['bndbox']['xmax']) / original_width)
                yMax = (float(obj['bndbox']['ymax']) / original_height)
                box = np.array(xMin, yMin, xMax, yMax)

                transformed_bbox = bbox_transform_pipeline(box)



        
