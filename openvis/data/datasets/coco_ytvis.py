import logging
import os

from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

logger = logging.getLogger(__name__)

COCO_TO_YTVIS_2019 = {
    1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18, 21:15, 22:32, 23:20, 24:30, 25:22, 35:33, 36:33, 41:5, 42:27, 43:40
}
COCO_TO_YTVIS_2021 = {
    1:26, 2:23, 3:5, 4:23, 5:1, 7:36, 8:37, 9:4, 16:3, 17:6, 18:9, 19:19, 21:7, 22:12, 23:2, 24:40, 25:18, 34:14, 35:31, 36:31, 41:29, 42:33, 43:34
}

COCO_TO_OVIS = {
    1:1, 2:21, 3:25, 4:22, 5:23, 6:25, 8:25, 9:24, 17:3, 18:4, 19:5, 20:6, 21:7, 22:8, 23:9, 24:10, 25:11, 
}



# ==== Predefined splits for COCO Video ===========
_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/coco2ovis_val.json"),
}


def register_all_coco_ytvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# Assume pre-defined datasets live in `./datasets`.
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_ytvis(_root)
