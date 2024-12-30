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

from openvis.data.ytvis import load_ytvis_json
from openvis.data.ovis import _PREDEFINED_SPLITS_OVIS


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
            'score': record['score'],
            'segmentations': record['segmentations'],
            'bboxes': [None] * len(record['segmentations']),
            'areas': areas,
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


def mask2box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return int(cmin), int(rmin), int(cmax), int(rmax)


if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ovis_val")

    dataset_image_root, dataset_json_file = [os.path.join('./datasets', x) for x in _PREDEFINED_SPLITS_OVIS["ovis_val"]]

    dirname = "brownian_ovis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    image_root = dataset_image_root

    pred_json_file = "./work_dirs/openvoc_lvvis/brownian_online_R50_bs8_12000st_ovis/eval/inference/results.json"
    pred_json_file = './results.json'
    gt_json_file = 'datasets/ovis/annotations_valid.json'

    if not os.path.exists('./ovis_pseudo_gt.json'):
        dataset_json = json.load(open(gt_json_file, 'r'))
        dataset_json["annotations"] = make_valid_annotations(json.load(open(pred_json_file, 'r')))
        json_file = os.path.join('./', 'ovis_pseudo_gt.json')
        json.dump(dataset_json, open(json_file, 'w'))

    pred_dicts = load_ytvis_json('./ovis_pseudo_gt.json', image_root, dataset_name="ovis_val")
    gt_dicts = load_ytvis_json(gt_json_file, image_root, dataset_name="ovis_val")
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

    colors = ['#A7EDE7', '#45CFDD', '#6527BE', '#A7EDE7', '#45CFDD', '#9681EB', '#6527BE']
    gamma = 0.7
    alpha = 0.3
    beta = 1.0

    for pd, gt in zip(pred_dicts, gt_dicts):

        vid_name = gt["file_names"][0].split('/')[-2]
        length = gt['length']

        if length < 50:
            continue

        # if vid_name not in ['e568ca41', '1ef6cb7b', 'd26036cd', '768c5810', '69398c01', '30446667']:
        #     continue

        if vid_name not in ['30446667']:
            continue

        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)

        for idx, file_name in enumerate(pd["file_names"]):
            img = np.array(Image.open(file_name))
            image_shape = img.shape[:2]

            frame_anno = extract_frame_dic(pd, idx)
            frame_anno = [x for x in frame_anno['annotations'] if x['score'] > 0.3]

            print(frame_anno)

            if len(frame_anno) == 0:
                break

            cats, masks = annotations_to_instances(frame_anno, image_shape)
            ids = [x['id'] for x in frame_anno]

            # new_mask = mask_util.decode(pd_segms[pd['annotations'][idx][1]['id']][idx])

            file_name = os.path.splitext(os.path.join(dirname, vid_name, file_name.split('/')[-1]))[0]

            # Image.fromarray(img).save(file_name + '_img.png')

            bimg = img.copy()

            valid_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for i in range(len(masks)):
                cat, mask, id = cats[i], masks[i], ids[i]

                timg = bimg.copy()

                # Image.fromarray(img).save(file_name + '.png')

                mask, gt_bound = generate_bound(mask)
                img[mask > 0] = (1 - alpha) * img[mask > 0, :] + alpha * np.array(hextorgb(colors[0]))
                img[gt_bound > 0, :] = (1 - beta) * img[gt_bound > 0, :] + (beta) * np.array(hextorgb(colors[0]))

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
