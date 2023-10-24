# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .ytvis import load_ytvis_json

"""
This file contains functions to parse UVO dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["register_uvo_instances"]

UVO_CATEGORIES = [
    {'color': [58, 186, 34], 'isthing': 1, 'id': 1, 'name': 'person'},
    {'color': [17, 100, 54], 'isthing': 1, 'id': 2, 'name': 'bicycle'},
    {'color': [210, 30, 182], 'isthing': 1, 'id': 3, 'name': 'car'},
    {'color': [61, 222, 3], 'isthing': 1, 'id': 4, 'name': 'motorcycle'},
    {'color': [155, 43, 13], 'isthing': 1, 'id': 5, 'name': 'airplane'},
    {'color': [180, 236, 196], 'isthing': 1, 'id': 6, 'name': 'bus'},
    {'color': [71, 184, 142], 'isthing': 1, 'id': 7, 'name': 'train'},
    {'color': [85, 6, 148], 'isthing': 1, 'id': 8, 'name': 'truck'},
    {'color': [244, 227, 239], 'isthing': 1, 'id': 9, 'name': 'boat'},
    {'color': [69, 21, 160], 'isthing': 1, 'id': 10, 'name': 'traffic light'},
    {'color': [182, 64, 108], 'isthing': 1, 'id': 11, 'name': 'fire hydrant'},
    {'color': [122, 44, 197], 'isthing': 1, 'id': 13, 'name': 'stop sign'},
    {'color': [79, 168, 10], 'isthing': 1, 'id': 14, 'name': 'parking meter'},
    {'color': [129, 92, 115], 'isthing': 1, 'id': 15, 'name': 'bench'},
    {'color': [129, 16, 165], 'isthing': 1, 'id': 16, 'name': 'bird'},
    {'color': [209, 241, 81], 'isthing': 1, 'id': 17, 'name': 'cat'},
    {'color': [130, 53, 195], 'isthing': 1, 'id': 18, 'name': 'dog'},
    {'color': [13, 108, 220], 'isthing': 1, 'id': 19, 'name': 'horse'},
    {'color': [199, 48, 4], 'isthing': 1, 'id': 20, 'name': 'sheep'},
    {'color': [94, 201, 242], 'isthing': 1, 'id': 21, 'name': 'cow'},
    {'color': [179, 230, 94], 'isthing': 1, 'id': 22, 'name': 'elephant'},
    {'color': [61, 208, 187], 'isthing': 1, 'id': 23, 'name': 'bear'},
    {'color': [233, 84, 39], 'isthing': 1, 'id': 24, 'name': 'zebra'},
    {'color': [114, 141, 178], 'isthing': 1, 'id': 25, 'name': 'giraffe'},
    {'color': [169, 144, 201], 'isthing': 1, 'id': 27, 'name': 'backpack'},
    {'color': [49, 246, 82], 'isthing': 1, 'id': 28, 'name': 'umbrella'},
    {'color': [130, 188, 44], 'isthing': 1, 'id': 31, 'name': 'handbag'},
    {'color': [121, 211, 18], 'isthing': 1, 'id': 32, 'name': 'tie'},
    {'color': [75, 197, 253], 'isthing': 1, 'id': 33, 'name': 'suitcase'},
    {'color': [91, 178, 27], 'isthing': 1, 'id': 34, 'name': 'frisbee'},
    {'color': [30, 241, 247], 'isthing': 1, 'id': 35, 'name': 'skis'},
    {'color': [179, 90, 34], 'isthing': 1, 'id': 36, 'name': 'snowboard'},
    {'color': [101, 198, 162], 'isthing': 1, 'id': 37, 'name': 'sports ball'},
    {'color': [173, 148, 16], 'isthing': 1, 'id': 38, 'name': 'kite'},
    {'color': [102, 136, 200], 'isthing': 1, 'id': 39, 'name': 'baseball bat'},
    {'color': [84, 180, 65], 'isthing': 1, 'id': 40, 'name': 'baseball glove'},
    {'color': [142, 117, 28], 'isthing': 1, 'id': 41, 'name': 'skateboard'},
    {'color': [137, 46, 249], 'isthing': 1, 'id': 42, 'name': 'surfboard'},
    {'color': [77, 6, 213], 'isthing': 1, 'id': 43, 'name': 'tennis racket'},
    {'color': [42, 7, 78], 'isthing': 1, 'id': 44, 'name': 'bottle'},
    {'color': [65, 129, 238], 'isthing': 1, 'id': 46, 'name': 'wine glass'},
    {'color': [94, 109, 70], 'isthing': 1, 'id': 47, 'name': 'cup'},
    {'color': [59, 13, 157], 'isthing': 1, 'id': 48, 'name': 'fork'},
    {'color': [155, 13, 162], 'isthing': 1, 'id': 49, 'name': 'knife'},
    {'color': [169, 202, 74], 'isthing': 1, 'id': 50, 'name': 'spoon'},
    {'color': [174, 122, 208], 'isthing': 1, 'id': 51, 'name': 'bowl'},
    {'color': [33, 7, 237], 'isthing': 1, 'id': 52, 'name': 'banana'},
    {'color': [81, 7, 146], 'isthing': 1, 'id': 53, 'name': 'apple'},
    {'color': [66, 91, 110], 'isthing': 1, 'id': 54, 'name': 'sandwich'},
    {'color': [195, 116, 129], 'isthing': 1, 'id': 55, 'name': 'orange'},
    {'color': [61, 112, 59], 'isthing': 1, 'id': 56, 'name': 'broccoli'},
    {'color': [141, 170, 129], 'isthing': 1, 'id': 57, 'name': 'carrot'},
    {'color': [54, 136, 146], 'isthing': 1, 'id': 58, 'name': 'hot dog'},
    {'color': [141, 33, 230], 'isthing': 1, 'id': 59, 'name': 'pizza'},
    {'color': [40, 134, 168], 'isthing': 1, 'id': 60, 'name': 'donut'},
    {'color': [5, 159, 121], 'isthing': 1, 'id': 61, 'name': 'cake'},
    {'color': [56, 74, 56], 'isthing': 1, 'id': 62, 'name': 'chair'},
    {'color': [15, 121, 179], 'isthing': 1, 'id': 63, 'name': 'couch'},
    {'color': [178, 208, 167], 'isthing': 1, 'id': 64, 'name': 'potted plant'},
    {'color': [120, 191, 110], 'isthing': 1, 'id': 65, 'name': 'bed'},
    {'color': [1, 126, 78], 'isthing': 1, 'id': 67, 'name': 'dining table'},
    {'color': [212, 139, 228], 'isthing': 1, 'id': 70, 'name': 'toilet'},
    {'color': [197, 233, 118], 'isthing': 1, 'id': 72, 'name': 'tv'},
    {'color': [8, 234, 30], 'isthing': 1, 'id': 73, 'name': 'laptop'},
    {'color': [93, 226, 16], 'isthing': 1, 'id': 74, 'name': 'mouse'},
    {'color': [235, 37, 210], 'isthing': 1, 'id': 75, 'name': 'remote'},
    {'color': [179, 190, 252], 'isthing': 1, 'id': 76, 'name': 'keyboard'},
    {'color': [73, 141, 136], 'isthing': 1, 'id': 77, 'name': 'cell phone'},
    {'color': [75, 12, 39], 'isthing': 1, 'id': 78, 'name': 'microwave'},
    {'color': [58, 44, 36], 'isthing': 1, 'id': 79, 'name': 'oven'},
    {'color': [144, 186, 206], 'isthing': 1, 'id': 80, 'name': 'toaster'},
    {'color': [81, 161, 103], 'isthing': 1, 'id': 81, 'name': 'sink'},
    {'color': [35, 33, 17], 'isthing': 1, 'id': 82, 'name': 'refrigerator'},
    {'color': [53, 51, 232], 'isthing': 1, 'id': 84, 'name': 'book'},
    {'color': [228, 103, 76], 'isthing': 1, 'id': 85, 'name': 'clock'},
    {'color': [39, 170, 239], 'isthing': 1, 'id': 86, 'name': 'vase'},
    {'color': [34, 137, 194], 'isthing': 1, 'id': 87, 'name': 'scissors'},
    {'color': [45, 179, 173], 'isthing': 1, 'id': 88, 'name': 'teddy bear'},
    {'color': [222, 125, 174], 'isthing': 1, 'id': 89, 'name': 'hair drier'},
    {'color': [125, 178, 223], 'isthing': 1, 'id': 90, 'name': 'toothbrush'},
    {'color': [154, 16, 69], 'isthing': 1, 'id': 91, 'name': 'other'},
]


def _get_uvo_instances_meta():
    thing_ids = [k["id"] for k in UVO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in UVO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 81, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 80]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in UVO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_uvo_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in UVO's json annotation format for
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


# ==== Predefined splits for UVO ===========
_PREDEFINED_SPLITS_UVO = {
    "uvo_train": ("uvo/uvo_videos_dense_frames",
                  "uvo/VideoDenseSet/UVO_video_train_dense_with_label.json"),
    "uvo_val":   ("uvo/uvo_videos_dense_frames",
                  "uvo/VideoDenseSet/UVO_video_val_dense_with_label.json"),
}


def register_all_uvo(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_uvo_instances(
            key,
            _get_uvo_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# Assume pre-defined datasets live in `./datasets`.
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_uvo(_root)


if __name__ == "__main__":
    """
    Test the UVO json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    import numpy as np
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("uvo_train")

    json_file = "./datasets/uvo/DenseVideoSet/UVO_video_train_dense_with_label.json"
    image_root = "./datasets/uvo/uvo_video_dense_frames/"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="uvo_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "uvo-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
