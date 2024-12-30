import os
import sys
from tqdm import tqdm

import json
import numpy as np
from skimage import morphology
from skimage.morphology import remove_small_objects
from pycocotools import mask as mask_util

from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

sys.path.append('./')

from openvis.data.datasets.ytvis import load_ytvis_json, _PREDEFINED_SPLITS_YTVIS_2019
from openvis.data.datasets.burst import load_burst_json, _PREDEFINED_SPLITS_BURST

def make_valid_annotations(records):
    annotations = []
    idx = 0
    for record in tqdm(records):
        # areas = []
        # for segm in record['segmentations']:
        #     area = int(np.sum(mask_util.decode(segm)))
        #     areas.append(None if area == 0 else area)
        idx += 1

        anno = {
            'video_id': record['video_id'],
            'iscrowd': 0,
            'id': idx,
            'category_id': record['category_id'],
            'segmentations': record['segmentations'],
            'bboxes': [[0., 0., 0., 0.]] * len(record['segmentations']),
            'areas': [0] * len(record['segmentations'])
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

    dataset_image_root, dataset_json_file = [os.path.join('./datasets', x) for x in _PREDEFINED_SPLITS_YTVIS_2019["ytvis_2019_train"]]

    gt_json_file = 'datasets/lvvis/val_ytvis_style.json'

    # 
    pred_json_file = "./cvpr2024/density/swin_san_online_best_bta.json"
    lens_dst_path = 'cvpr2024/density/ytvis_2019_mean_brow_lens.npy'
    ious_dst_path = 'cvpr2024/density/ytvis_2019_mean_brow_ious.npy'

    #
    # pred_json_file = "./cvpr2024/density/san_online_worse_baseline.json"
    # lens_dst_path = 'cvpr2024/density/ytvis_2019_mean_dire_lens.npy'
    # ious_dst_path = 'cvpr2024/density/ytvis_2019_mean_dire_ious.npy'

    coco_res = json.load(open(pred_json_file, 'r'))

    # When evaluating mask AP, if the results contain bbox, cocoapi will
    # use the box area as the area of the instance, instead of the mask area.
    # This leads to a different definition of small/medium/large.
    # We remove the bbox field to let mask AP use mask area.
    for c in coco_res:
        c.pop("bbox", None)

    from openvis.data.evals.ytvos import YTVOS
    from openvis.data.evals.ytvis_eval import YTVOSeval
    coco_gt = YTVOS(gt_json_file)
    coco_dt = coco_gt.loadRes(pred_json_file)
    coco_eval = YTVOSeval(coco_gt, coco_dt)

    # For COCO, the default max_dets_per_image is [1, 10, 100].
    max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    coco_eval.params.maxDets = max_dets_per_image

    coco_eval.evaluate()

    _ious = coco_eval.ious
    _gts = coco_eval._gts
    _dts = coco_eval._dts

    lens = []
    ious = []
    keys = list(_ious.keys())
    for key in keys:
        if len(_gts[key]) == 0:
            continue
        if len(_dts[key]) == 0:
            iou = np.zeros(len(_gts[key]))
        else:
            iou = _ious[key].mean(axis=0)
        ious.extend(iou)
        lens.extend([len(list(filter(None, x['areas']))) for x in _gts[key]])

    np.save(lens_dst_path, np.array(lens))
    np.save(ious_dst_path, np.array(ious))
