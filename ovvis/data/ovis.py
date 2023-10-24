# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .ytvis import load_ytvis_json

"""
This file contains functions to parse OVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["register_ovis_instances"]


OVIS_CATEGORIES = [
    {'color': [98, 163, 136], 'isthing': 1, 'id': 1, 'name': 'Person'},
    {'color': [99, 225, 61], 'isthing': 1, 'id': 2, 'name': 'Bird'},
    {'color': [140, 188, 60], 'isthing': 1, 'id': 3, 'name': 'Cat'},
    {'color': [64, 20, 38], 'isthing': 1, 'id': 4, 'name': 'Dog'},
    {'color': [245, 158, 43], 'isthing': 1, 'id': 5, 'name': 'Horse'},
    {'color': [153, 127, 26], 'isthing': 1, 'id': 6, 'name': 'Sheep'},
    {'color': [152, 223, 242], 'isthing': 1, 'id': 7, 'name': 'Cow'},
    {'color': [98, 215, 152], 'isthing': 1, 'id': 8, 'name': 'Elephant'},
    {'color': [214, 179, 173], 'isthing': 1, 'id': 9, 'name': 'Bear'},
    {'color': [204, 22, 1], 'isthing': 1, 'id': 10, 'name': 'Zebra'},
    {'color': [186, 176, 37], 'isthing': 1, 'id': 11, 'name': 'Giraffe'},
    {'color': [120, 166, 90], 'isthing': 1, 'id': 12, 'name': 'Poultry'},
    {'color': [118, 154, 40], 'isthing': 1, 'id': 13, 'name': 'Giant_panda'},
    {'color': [135, 129, 93], 'isthing': 1, 'id': 14, 'name': 'Lizard'},
    {'color': [69, 178, 54], 'isthing': 1, 'id': 15, 'name': 'Parrot'},
    {'color': [73, 40, 114], 'isthing': 1, 'id': 16, 'name': 'Monkey'},
    {'color': [153, 200, 46], 'isthing': 1, 'id': 17, 'name': 'Rabbit'},
    {'color': [48, 20, 213], 'isthing': 1, 'id': 18, 'name': 'Tiger'},
    {'color': [79, 61, 56], 'isthing': 1, 'id': 19, 'name': 'Fish'},
    {'color': [14, 93, 200], 'isthing': 1, 'id': 20, 'name': 'Turtle'},
    {'color': [126, 14, 7], 'isthing': 1, 'id': 21, 'name': 'Bicycle'},
    {'color': [61, 93, 237], 'isthing': 1, 'id': 22, 'name': 'Motorcycle'},
    {'color': [75, 129, 223], 'isthing': 1, 'id': 23, 'name': 'Airplane'},
    {'color': [53, 98, 192], 'isthing': 1, 'id': 24, 'name': 'Boat'},
    {'color': [155, 74, 85], 'isthing': 1, 'id': 25, 'name': 'Vehical'},
]

def _get_ovis_instances_meta():
    thing_ids = [k["id"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 25, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_ovis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in OVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )


# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                   "ovis/annotations_train.json"),
    "ovis_val":   ("ovis/valid",
                   "ovis/annotations_valid.json"),
    "ovis_test":  ("ovis/test",
                   "ovis/annotations_test.json"),
}


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ovis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# Assume pre-defined datasets live in `./datasets`.
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ovis(_root)
