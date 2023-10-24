import os
import json
from tqdm import tqdm

import imagesize
import numpy as np
from pycocotools import mask as mask_util


def formatting(src_json, dst_json):
    uvo_json = json.load(open(src_json, 'r'))

    vids = uvo_json['videos']
    cats = list(set(uvo_json['categories']))
    anns = uvo_json['annotations']

    print(cats)
    exit(0)

    ytvis_json = {
        'videos': vids,
        'categories': cats,
        'annotations': anns,
    }

    json.dump(ytvis_json, open(dst_json, 'w'))


# formatting train annotations
# src_json = 'datasets/uvo/VideoDenseSet/UVO_video_train_dense_with_label.json'
# dst_json = 'datasets/uvo/train_ytvis_style.json'

# print("Start Converting train annotation:")
# formatting(src_json, dst_json)

src_json = 'datasets/uvo/VideoDenseSet/UVO_video_val_dense_with_label.json'
dst_json = 'datasets/uvo/val_ytvis_style.json'

print("Start Converting val annotation:")
formatting(src_json, dst_json)
