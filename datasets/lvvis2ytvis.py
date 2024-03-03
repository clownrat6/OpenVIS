import os
import json
from tqdm import tqdm

import imagesize
import numpy as np
from pycocotools import mask as mask_util


def mask2box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return int(cmin), int(rmin), int(cmax), int(rmax)


def videos_formatting(vids, img_folder):
    error_vids = {}
    new_vids = []
    for vid in tqdm(vids):
        h, w = vid['height'], vid['width']
        whs = [imagesize.get(os.path.join(img_folder, x)) for x in vid['file_names']]
        invalid_frame_ids = []
        ifs = []
        for f_id, (rw, rh) in enumerate(whs):
            if rh != h or rw != w:
                invalid_frame_ids.append(f_id)
                ifs.append(vid['file_names'][f_id])

        if len(invalid_frame_ids) != 0:
            error_vids[vid['id']] = invalid_frame_ids
            vid['file_names'] = vid['file_names'][:invalid_frame_ids[0]]
            vid['length'] = len(vid['file_names'])

        new_vids.append(vid)

    return new_vids, error_vids


def annotations_formatting(anns, err_vids):
    new_anns = []
    for ann in tqdm(anns):
        if ann['video_id'] in err_vids:
            vid_id = ann['video_id']
            err_st_idx = err_vids[vid_id][0]
            ann['segmentations'] = ann['segmentations'][:err_st_idx]
            ann['areas'] = ann['areas'][:err_st_idx]

        bboxes = []
        for segm in ann['segmentations']:
            if segm is None:
                bboxes.append(None)
            else:
                mask = mask_util.decode(segm)
                if np.sum(mask) == 0:
                    bboxes.append([0, 0, 0, 0])
                    continue
                x1, y1, x2, y2 = mask2box(mask)
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
                bboxes.append([x, y, w, h])
        new_ann = {}
        new_ann.update(ann)
        new_ann['bboxes'] = bboxes
        new_anns.append(new_ann)

    return new_anns


def formatting(src_json, dst_json, img_folder):
    lvvis_json = json.load(open(src_json, 'r'))

    vids, err_vids = videos_formatting(lvvis_json['videos'], img_folder)
    cats = lvvis_json['categories']
    anns = annotations_formatting(lvvis_json['annotations'], err_vids)

    ytvis_json = {
        'videos': vids,
        'categories': cats,
        'annotations': anns,
    }

    json.dump(ytvis_json, open(dst_json, 'w'))


def construct_json(folder,cats):
    vid_list = os.listdir(folder)

    total_json = {
        'categories': cats,
    }

    vids = []
    for idx, vid_folder in enumerate(tqdm(vid_list)):
        frame_list = os.listdir(os.path.join(folder, vid_folder))
        frame_paths = [os.path.join(vid_folder, x) for x in frame_list]
        w, h = imagesize.get(os.path.join(folder, frame_paths[0]))
        vids.append({
            'id': idx,
            'height': h,
            'width': w,
            'length': len(frame_paths),
            'file_names': frame_paths,
        })

    total_json['videos'] = vids

    return total_json

# formatting train annotations
img_folder = 'datasets/lvvis/train/JPEGImages'
src_json = 'datasets/lvvis/train_instances.json'
dst_json = 'datasets/lvvis/train_ytvis_style.json'

print("Start Converting train annotation:")
formatting(src_json, dst_json, img_folder)

img_folder = 'datasets/lvvis/val/JPEGImages'
src_json = 'datasets/lvvis/val_instances.json'
dst_json = 'datasets/lvvis/val_ytvis_style.json'

print("Start Converting val annotation:")
formatting(src_json, dst_json, img_folder)

img_folder = 'datasets/lvvis/test/JPEGImages'
src_json = 'datasets/lvvis/val_instances.json'
dst_json = 'datasets/lvvis/test_ytvis_style.json'

tmp = json.load(open(src_json, 'r'))
res = construct_json(img_folder, tmp['categories'])
json.dump(res, open(dst_json, 'w'))

