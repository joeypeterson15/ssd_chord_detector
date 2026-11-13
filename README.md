# SSD Chord Detector

SSD300 detector for guitar photos. Each fretting finger is its own box, labeled with a note letter, and the fretboard is a separate class. The chord is the set of notes that come back.

The checkpoint in `ssd_chord_model.pth` was trained on 71 annotated photos. On a clean diagram it can place A, C, and E on an A-minor shape and draw the fretboard. Phone photos are less consistent, which is expected for a set this small.

## Layout

| Path | Role |
|---|---|
| `chord_index.py` | Dataset, augmentation, and the 25-epoch training loop. Saves the best validation loss to `ssd_chord_model.pth`. |
| `saved_model_inference.py` | Loads that checkpoint and draws detections with matplotlib (`torch_snippets.show`). |
| `detect.py` | Resize to 300×300, normalize, run the network, scale boxes back to the original image. |
| `ssd_utils/model.py` | SSD300: VGG-16 backbone, auxiliary feature maps, prediction heads, and MultiBox loss. |
| `ssd_utils/utils.py` | Box encoding, IoU, and the fully-connected-to-convolution weight conversion. |
| `chord_archive/` | Training images and Pascal VOC XML. Gitignored; keep it next to the scripts. |
| `bus_truck_index.py` | Earlier bus/truck tutorial this training loop was adapted from. |

`chord_shape` boxes are in the XML and are not used. The label list also includes F and A#, which do not appear in the 71 photos.

## Setup

Python 3.12. From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib albumentations xmltodict pandas numpy scikit-learn torch-snippets pillow ipython
```

## Run inference

```bash
python -u saved_model_inference.py
```

That walks a slice of `chord_archive/images_train` and opens a matplotlib window per photo. Close a window to see the next one. The script applies a small random affine and brightness change before detection, so a few frames can come back as a single background box.

## Train

```bash
python chord_index.py
```

This trains for 25 epochs on CPU or CUDA, writes `ssd_chord_model.pth` whenever validation loss improves, then runs the same detection view.

## Model, briefly

Input is 300×300. A VGG-16 trunk plus four extra convolutions produce six feature maps, from 38×38 down to 1×1. Each cell on those maps owns several default boxes (8732 total) at different scales and aspect ratios. The network predicts a class score and an encoded offset for every default box in one pass. Training is MultiBox loss: cross-entropy with hard-negative mining, plus L1 on the box offsets for priors that overlap a ground-truth box.
