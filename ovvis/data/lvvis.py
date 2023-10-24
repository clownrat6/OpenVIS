# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC
import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog

from .ytvis import load_ytvis_json
from .lvvis_cat import LVVIS_CATEGORIES

logger = logging.getLogger(__name__)

__all__ = ["register_lvvis_instances"]


def _get_lvvis_instances_meta():
    thing_ids = [k["id"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1196, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_lvvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
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


# ==== Predefined splits for LVVIS ===========
_PREDEFINED_SPLITS_LVVIS = {
    "lvvis_train": ("lvvis/train/JPEGImages",
                    "lvvis/train_ytvis_style.json"),
    "lvvis_val":   ("lvvis/val/JPEGImages",
                    "lvvis/val_ytvis_style.json"),
}


def register_all_lvvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_lvvis_instances(
            key,
            _get_lvvis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# Assume pre-defined datasets live in `./datasets`.
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lvvis(_root)
