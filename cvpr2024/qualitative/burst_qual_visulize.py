import os
import sys
import json
from tqdm import tqdm

import numpy as np
from PIL import Image
from skimage import morphology
from skimage.morphology import remove_small_objects
from pycocotools import mask as mask_util

from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

sys.path.append('./')

from openvis.data.burst import load_burst_json, _PREDEFINED_SPLITS_BURST, ALL_BURST_CATEGORIES
from openvis.data.lvvis_cat import LVVIS_CATEGORIES


def make_valid_annotations(records):
    annotations = []
    idx = 0
    for record in tqdm(records):
        areas = []
        for segm in record['segmentations']:
            area = int(np.sum(mask_util.decode(segm)))
            areas.append(None if area == 0 else area)
        idx += 1
        anno = {
            'video_id': record['video_id'],
            'iscrowd': 0,
            'id': idx,
            'category_id': record['category_id'],
            'segmentations': record['segmentations'],
            'bboxes': [None] * len(record['segmentations']),
            'areas': areas
        }
        annotations.append(anno)

    return annotations


def iou_seq(d_seq, g_seq):
    i = .0
    u = .0
    for d, g in zip(d_seq, g_seq):
        if d and g:
            i += mask_util.area(mask_util.merge([d, g], True))
            u += mask_util.area(mask_util.merge([d, g], False))
        elif not d and g:
            u += mask_util.area(g)
        elif d and not g:
            u += mask_util.area(d)
    if not u > .0:
        print("Mask sizes in video may not match!")
    iou = i / u if u > .0 else .0
    return iou


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    from detectron2.structures import polygons_to_bitmask
    classes = [int(obj["category_id"]) for obj in annos]

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image_size))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(segm))
                )
            # torch.from_numpy does not support array with negative stride.
            masks = [np.ascontiguousarray(x) for x in masks]

    return classes, masks


def hextorgb(num):
    num = num[1:]
    return (int(num[:2], 16), int(num[2:4], 16), int(num[4:6], 16))


def generate_bound(mask):
    mask = remove_small_objects(mask, min_size=20)
    # mask = morphology.dilation(mask, footprint=morphology.diamond(2))
    bound = morphology.dilation(
        mask, footprint=morphology.diamond(1)) & (~morphology.erosion(mask, footprint=morphology.diamond(3)))

    return mask, bound


known_list = [4, 13, 1038, 544, 1057, 34, 35, 36, 41, 45, 58, 60, 579, 1091, 1097, 1099, 78, 79, 81, 91, 1115,
                1117, 95, 1122, 99, 1132, 621, 1135, 625, 118, 1144, 126, 642, 1155, 133, 1162, 139, 154, 174, 185,
                699, 1215, 714, 717, 1229, 211, 729, 221, 229, 747, 235, 237, 779, 276, 805, 299, 829, 852, 347,
                371, 382, 896, 392, 926, 937, 428, 429, 961, 452, 979, 980, 982, 475, 480, 993, 1001, 502, 1018]


def filter_element(old_list, filter_value):
    return [x for x in old_list if x != filter_value]


if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("burst_val")

    known_classes = [meta.thing_dataset_id_to_contiguous_id[x] for x in known_list if x in meta.thing_dataset_id_to_contiguous_id]

    burst_classes = [x['name'] for x in ALL_BURST_CATEGORIES]
    lvvis_classes = [x['name'] for x in LVVIS_CATEGORIES]

    novel_classes = list(set(burst_classes) - set(lvvis_classes).intersection(set(burst_classes)))
    novel_classes = [meta.name_to_contiguous_id[x] for x in novel_classes]

    id_to_classes = {v:k for k,v in meta.name_to_contiguous_id.items()}

    dataset_image_root, dataset_json_file = [os.path.join('./datasets', x) for x in _PREDEFINED_SPLITS_BURST["burst_val"]]

    dirname = "brownian_burst-data-vis"
    os.makedirs(dirname, exist_ok=True)

    image_root = dataset_image_root

    pred_json_file = "./work_dirs/openvoc_lvvis/brownian_online_R50_bs8_12000st/inference/results.json"
    gt_json_file = 'datasets/burst/annotations/val/all_classes.json'

    pred_dicts = load_burst_json(pred_json_file, image_root, dataset_name="burst_val")
    gt_dicts = load_burst_json(gt_json_file, image_root, dataset_name="burst_val")
    logger.info("Done loading {} samples.".format(len(pred_dicts)))

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    def introduce(dict, key, value):
        if key not in dict:
            dict[key] = [value]
        else:
            dict[key].append(value)

    colors = ['#45CFDD', '#9681EB', '#6527BE', '#A7EDE7', '#45CFDD', '#9681EB', '#6527BE']
    gamma = 0.7
    alpha = 0.3
    beta = 1.0

    for pd, gt in zip(pred_dicts, gt_dicts):
        pd_cat_ids = {}
        gt_cat_ids = {}
        pd_segms = {}
        gt_segms = {}
        pd_track_ids = []
        gt_track_ids = []
        for pds, gts in zip(pd['annotations'], gt['annotations']):
            for pdd in pds:
                pd_track_ids.append(pdd['id'])
            for gtt in gts:
                gt_track_ids.append(gtt['id'])
        pd_track_ids = list(set(pd_track_ids))
        gt_track_ids = list(set(gt_track_ids))

        for pds, gts in zip(pd['annotations'], gt['annotations']):
            accu_ids = []
            for pdd in pds:
                accu_ids.append(pdd['id'])
                introduce(pd_cat_ids, pdd['id'], pdd['category_id'])
                introduce(pd_segms, pdd['id'], pdd['segmentation'])
            for pd_id in pd_track_ids:
                if pd_id not in accu_ids:
                    introduce(pd_cat_ids, pd_id, None)
                    introduce(pd_segms, pd_id, None)

            accu_ids = []
            for gtt in gts:
                accu_ids.append(gtt['id'])
                introduce(gt_cat_ids, gtt['id'], gtt['category_id'])
                introduce(gt_segms, gtt['id'], gtt['segmentation'])
            for gt_id in gt_track_ids:
                if gt_id not in accu_ids:
                    introduce(gt_cat_ids, gt_id, None)
                    introduce(gt_segms, gt_id, None)      

        ious = np.zeros((len(pd_track_ids), len(gt_track_ids)))
        for i, pd_id in enumerate(pd_track_ids):
            for j, gt_id in enumerate(gt_track_ids):
                ious[i, j] = iou_seq(pd_segms[pd_id], gt_segms[gt_id])

        vid_name = gt["file_names"][0].split('/')[-2]
        if vid_name not in ['Washing_hands_v_vvRlK1oeAow']:
            continue
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)

        # for idx, file_name in enumerate(gt["file_names"]):
        #     img = np.array(Image.open(file_name))
        #     image_shape = img.shape[:2]

        #     cats, masks = annotations_to_instances(gt['annotations'][idx], image_shape)
        #     cat, mask = cats[0], masks[0]

        #     file_name = os.path.splitext(os.path.join('./', file_name.split('/')[-1]))[0]

        #     mask, gt_bound = generate_bound(mask)
        #     # canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
        #     # canvas[mask > 0, :] = np.array(hextorgb(mask_color))
        #     # img = cv2.addWeighted(img, (1 - alpha), canvas, alpha, 1.0)
        #     img[mask == 0] = img[mask == 0, :] * 0.4
        #     # img[mask == 0] = (1 - gamma) * img[mask == 0, :] + gamma * np.array(hextorgb(bg_color))
        #     # img[mask > 0] = (1 - alpha) * img[mask > 0, :] + alpha * np.array(hextorgb(gt_color))
        #     # img[gt_bound > 0, :] = (1 - beta) * img[gt_bound > 0, :] + (beta) * np.array(hextorgb(gt_color))
        #     Image.fromarray(img).save(file_name + '_seg.png')

        #     # visualizer = Visualizer(img, metadata=meta)
        #     # vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
        #     # fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
        #     # vis.save(fpath)

        gt_lengths = [len(filter_element(gt_segms[gt_id], None)) for gt_id in gt_track_ids]
        _gt_segms = [gt_segms[gt_id] for gt_id in gt_track_ids]
        _gt_cat_ids = [filter_element(gt_cat_ids[gt_id], None)[0] for gt_id in gt_track_ids]

        valid_ids = []
        valid_pd_ids = []
        for idx, gt_id in enumerate(gt_track_ids):
            cat_id = filter_element(gt_cat_ids[gt_id], None)[0]
            if cat_id in novel_classes and ious.max(axis=0)[idx] > 0.6:
                # print(ious.max(axis=0)[idx], gt_lengths[idx], _gt_cat_ids[idx])
                valid_pd_ids.append(pd_track_ids[np.argmax(ious, axis=0)[idx]])
                valid_ids.append(gt_id)

        if len(valid_ids) == 0:
            continue

        for idx, file_name in enumerate(pd["file_names"]):
            img = np.array(Image.open(file_name))
            image_shape = img.shape[:2]

            file_name = os.path.splitext(os.path.join(dirname, vid_name, file_name.split('/')[-1]))[0]

            # Image.fromarray(img).save(file_name + '_img.png')

            bimg = img.copy()

            masks = [mask_util.decode(segm[idx]) if segm[idx] is not None else np.zeros(image_shape, dtype=np.uint8) for segm in _gt_segms]
            cats = _gt_cat_ids

            print(file_name, [id_to_classes[x] for x in cats if x in novel_classes])

            valid_mask = np.zeros(img.shape[:2], dtype=np.uint8)

            for i in range(len(masks)):
                cat, mask = cats[i], masks[i]

                timg = bimg.copy()

                # Image.fromarray(img).save(file_name + '.png')

                color_id = 1 if cat == 304 else 0

                mask, gt_bound = generate_bound(mask)
                img[mask > 0] = (1 - alpha) * img[mask > 0, :] + alpha * np.array(hextorgb(colors[color_id]))
                img[gt_bound > 0, :] = (1 - beta) * img[gt_bound > 0, :] + (beta) * np.array(hextorgb(colors[color_id]))

                valid_mask[mask > 0] = 1
                valid_mask[gt_bound > 0] = 1

                # timg[mask > 0] = (1 - alpha) * timg[mask > 0, :] + alpha * np.array(hextorgb(gt_colors[id_memory.index(id)]))
                # timg[gt_bound > 0, :] = (1 - beta) * timg[gt_bound > 0, :] + (beta) * np.array(hextorgb(gt_colors[id_memory.index(id)]))
                # timg[(mask == 0) * (gt_bound == 0)] = timg[(mask == 0) * (gt_bound == 0), :] * 0.5
                # Image.fromarray(timg).save(file_name + f'_{id}_seg.png')

            # canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
            # canvas[mask > 0, :] = np.array(hextorgb(mask_color))
            # img = cv2.addWeighted(img, (1 - alpha), canvas, alpha, 1.0)
            # img[mask == 0] = (1 - gamma) * img[mask == 0, :] + gamma * np.array(hextorgb(bg_color))
            img[valid_mask == 0] = img[valid_mask == 0, :] * 0.4
            Image.fromarray(img).save(file_name + '_seg.png')

        continue

        height, width = gt['height'], gt['width']

        for idx, file_name in enumerate(pd["file_names"]):
            img = np.array(Image.open(file_name))
            image_shape = img.shape[:2]

            cats, masks = annotations_to_instances(pd['annotations'][idx], image_shape)
            cat, mask = cats[1], masks[1]

            # new_mask = mask_util.decode(pd_segms[pd['annotations'][idx][1]['id']][idx])
            # print(np.allclose(mask, new_mask))
            # exit(0)

            file_name = os.path.splitext(os.path.join('./', file_name.split('/')[-1]))[0]

            mask, gt_bound = generate_bound(mask)
            # canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
            # canvas[mask > 0, :] = np.array(hextorgb(mask_color))
            # img = cv2.addWeighted(img, (1 - alpha), canvas, alpha, 1.0)
            img[mask == 0] = img[mask == 0, :] * 0.4
            # img[mask == 0] = (1 - gamma) * img[mask == 0, :] + gamma * np.array(hextorgb(bg_color))
            # img[mask > 0] = (1 - alpha) * img[mask > 0, :] + alpha * np.array(hextorgb(gt_color))
            # img[gt_bound > 0, :] = (1 - beta) * img[gt_bound > 0, :] + (beta) * np.array(hextorgb(gt_color))
            Image.fromarray(img).save(file_name + '_seg.png')

            # visualizer = Visualizer(img, metadata=meta)
            # vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            # fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            # vis.save(fpath)
