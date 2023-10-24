# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation

__all__ = ['BURSTDatasetMapper']


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    '''
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    '''
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has('gt_masks') and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances

def transform_instance_segmentation_annotations(
    annotation, transforms, image_size
):
    """
    Apply transforms to segmentation annotations of a single instance.

    It will use `transforms.apply_coords` for segmentation polygons.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image

    Returns:
        dict:
            the same input dict with fields "segmentation"
            transformed according to `transforms`.
    """
    from pycocotools import mask as mask_util

    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    return annotation


def segmentation_annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance segmentation annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance segmentation annotations 
            in one image, each element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_classes", "gt_masks", if they can 
            be obtained from `annos`. This is the format that builtin models expect.
    """
    from pycocotools import mask as mask_util
    from detectron2.structures import (
        BitMasks,
        Instances,
        PolygonMasks,
        polygons_to_bitmask,
    )

    target = Instances(image_size)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
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
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    return target


def _get_dummy_anno(num_classes):
    return {
        'iscrowd': 0,
        'category_id': num_classes,
        'id': -1,
        'segmentation': [np.array([0.0] * 6)]
    }


class BURSTDatasetMapper:
    '''
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    '''

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        dataset_name: str = 'burst_val',
    ):
        '''
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        '''
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        self.dataset_name           = dataset_name
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = 'training' if is_train else 'inference'
        logger.info(f'[DatasetMapper] Augmentations used in {mode}: {augmentations}')

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        if is_train:
            dataset_name = cfg.DATASETS.TRAIN[0]
        else:
            dataset_name = cfg.DATASETS.TEST[0]

        ret = {
            'is_train': is_train,
            'augmentations': augs,
            'image_format': cfg.INPUT.FORMAT,
            'use_instance_mask': cfg.MODEL.MASK_ON,
            'sampling_frame_num': sampling_frame_num,
            'sampling_frame_range': sampling_frame_range,
            'sampling_frame_shuffle': sampling_frame_shuffle,
            'num_classes': cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            'dataset_name': dataset_name
        }

        return ret

    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict['length']
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop('annotations', None)
        file_names = dataset_dict.pop('file_names', None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno['id'] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["dataset_name"] = self.dataset_name
        dataset_dict['image'] = []
        dataset_dict['instances'] = []
        dataset_dict['file_names'] = []
        for frame_idx in selected_idx:
            dataset_dict['file_names'].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict['image'].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # NOTE: requirement attribute example: {
            #     'width': 640,
            #     'height': 480,
            #     'seq_name': 'v_25685519b728afd746dfd1b2fe77c', 
            #     'dataset': 'YFCC100M',
            #     'image_paths': ['frame0751.jpg', 'frame0781.jpg', ..., 'frame1921.jpg'],
            #     'image': [torch.zeros((3, H, W)), ...],
            #     'instances': PolygonMask(),
            # }

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_segmentation_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get('iscrowd', 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno['id']]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno['id'] for _anno in sorted_annos]

            instances = segmentation_annotations_to_instances(sorted_annos, image_shape, mask_format='bitmask')
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has('gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict['instances'].append(instances)

        print(dataset_dict.keys())
        [print(x.shape) for x in dataset_dict["image"]]
        print(len(dataset_dict['instances']), len(dataset_dict['annotated_image_paths']))
        exit(0)

        return dataset_dict
