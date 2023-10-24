import os
import json
from tqdm import tqdm

import numpy as np
import mmcv
from pycocotools import mask as mask_util


def mask2box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return int(cmin), int(rmin), int(cmax), int(rmax)


def categories_formatting(categories):
    new_categories = [{'id': cat['id'], 'name': cat['name']} for cat in categories]
    return new_categories

def videos_formatting(sequences):
    # def inner_func(sequence):
    #     return {
    #         'width': sequence['width'],
    #         'height': sequence['height'],
    #         'length': len(sequence['annotated_image_paths']),
    #         'id': sequence['id'],
    #     }
    # return mmcv.track_parallel_progress(inner_func, sequences, 8)
    new_videos = []
    for sequence in tqdm(sequences):
        new_videos.append({
            'width': sequence['width'],
            'height': sequence['height'],
            'length': len(sequence['annotated_image_paths']),
            'id': sequence['id'],
            'file_names': [os.path.join(sequence['dataset'], sequence['seq_name'], x) for x in sequence['annotated_image_paths']],
        })
    
    return new_videos

def annotations_formatting(sequences):
    new_annotations = []
    inst_id = 0
    for sequence in tqdm(sequences):
        for track_id, cat_id in sequence['track_category_ids'].items():
            width = sequence['width']
            height = sequence['height']
            segmentations = []
            areas = []
            bboxes = []
            for segm in sequence['segmentations']:
                if track_id not in segm:
                    bboxes.append(None)
                    segmentations.append(None)
                    areas.append(None)
                else:
                    mask = mask_util.decode({'counts': segm[track_id]['rle'], 'size': [height, width]})
                    x1, y1, x2, y2 = mask2box(mask)
                    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
                    mask_rle = mask_util.encode(mask)
                    mask_rle['counts'] = mask_rle['counts'].decode("utf-8")
                    bboxes.append([x, y, w, h])
                    segmentations.append(mask_rle)
                    areas.append(int(np.sum(mask)))

            new_annotations.append({
                'width': sequence['width'],
                'height': sequence['height'],
                'category_id': cat_id,
                'video_id': sequence['id'],
                'id': inst_id,
                'areas': areas,
                'segmentations': segmentations,
                'bboxes': bboxes,
                'iscrowd': 0,
            })
            inst_id += 1
    
    return new_annotations


src_json = 'datasets/burst/annotations/val/all_classes.json'
dst_json = 'datasets/burst/val_ytvis.json'

burst_json = json.load(open(src_json, 'r'))

ytvis_json = {
    'videos': videos_formatting(burst_json['sequences']),
    'categories': categories_formatting(burst_json['categories']),
    'annotations': annotations_formatting(burst_json['sequences']),
}

json.dump(ytvis_json, open(dst_json, 'w'))