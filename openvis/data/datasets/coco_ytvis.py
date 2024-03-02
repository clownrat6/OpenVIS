import logging
import os

from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

logger = logging.getLogger(__name__)

COCO_TO_YTVIS_2019 = {1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18, 21:15, 22:32, 23:20, 24:30, 25:22, 35:33, 36:33, 41:5, 42:27, 43:40}
COCO_TO_YTVIS_2021 = {1:26, 2:23, 3:5, 4:23, 5:1, 7:36, 8:37, 9:4, 16:3, 17:6, 18:9, 19:19, 21:7, 22:12, 23:2, 24:40, 25:18, 34:14, 35:31, 36:31, 41:29, 42:33, 43:34}
COCO_TO_OVIS = {1:1, 2:21, 3:25, 4:22, 5:23, 6:25, 8:25, 9:24, 17:3, 18:4, 19:5, 20:6, 21:7, 22:8, 23:9, 24:10, 25:11}
YTVIS_2019_TO_COCO = {v: k for k, v in COCO_TO_YTVIS_2019.items()}
YTVIS_2021_TO_COCO = {v: k for k, v in COCO_TO_YTVIS_2021.items()}
OVIS_TO_COCO = {v: k for k, v in COCO_TO_OVIS.items()}


# ==== Predefined splits for COCO Video ===========
_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/coco2ovis_val.json"),
    "ytvis2019_train2coco": ("coco/train2017", "coco/ytvis2019_train2coco.json"),
    "ytvis2019_val2coco": ("coco/val2017", "coco/ytvis2019_val2coco.json"),
    "ytvis2021_train2coco": ("coco/train2017", "coco/ytvis2021_train2coco.json"),
    "ytvis2021_val2coco": ("coco/val2017", "coco/ytvis2021_val2coco.json"),
    "ovis_train2coco": ("coco/train2017", "coco/ovis_train2coco.json"),
    "ovis_val2coco": ("coco/val2017", "coco/ovis_val2coco.json"),
}

COCO_YTVIS_CATEGORIES = [
    {"color": [64, 198, 187], "isthing": 1, "id": 1, "name": "person"},
    {"color": [73, 245, 54], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [5, 81, 214], "isthing": 1, "id": 3, "name": "car"},
    {"color": [168, 140, 60], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [73, 25, 118], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [162, 203, 237], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [254, 190, 24], "isthing": 1, "id": 7, "name": "train"},
    {"color": [14, 145, 166], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [148, 66, 44], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 248, 199], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [33, 118, 47], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [250, 39, 187], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [95, 207, 32], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [56, 162, 76], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [16, 32, 57], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [77, 230, 28], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [10, 185, 48], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [32, 115, 137], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [224, 151, 51], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [251, 128, 247], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [235, 113, 37], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [0, 7, 226], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [54, 233, 134], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [215, 168, 56], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [167, 217, 69], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [99, 107, 29], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [61, 217, 215], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [138, 232, 176], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [250, 122, 235], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [182, 226, 32], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [221, 229, 68], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [96, 168, 12], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [219, 219, 163], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [125, 4, 158], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [154, 91, 155], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [210, 177, 120], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [103, 220, 83], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [200, 26, 219], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [68, 237, 106], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [27, 24, 195], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [73, 155, 190], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [32, 208, 13], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [139, 69, 168], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [85, 228, 162], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [144, 82, 227], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 164, 77], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [124, 121, 193], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [142, 11, 193], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [225, 99, 97], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [73, 16, 182], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [52, 238, 140], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [129, 119, 128], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [216, 207, 163], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [192, 67, 184], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [106, 227, 136], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [141, 65, 220], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [19, 215, 213], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [247, 174, 202], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [184, 165, 49], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [208, 145, 93], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [1, 137, 32], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [234, 209, 104], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [143, 236, 241], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [115, 104, 212], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [238, 65, 77], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [4, 198, 51], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [186, 209, 78], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [5, 154, 98], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [20, 130, 125], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [244, 16, 35], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [52, 183, 157], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [63, 38, 57], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [200, 172, 154], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [102, 207, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [85, 1, 201], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [18, 222, 247], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [47, 242, 149], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [161, 159, 207], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [128, 243, 117], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [76, 221, 10], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [245, 152, 44], "isthing": 1, "id": 2000, "name": "giant_panda"},
    {"color": [129, 221, 75], "isthing": 1, "id": 2001, "name": "lizard"},
    {"color": [221, 96, 247], "isthing": 1, "id": 2002, "name": "parrot"},
    {"color": [178, 229, 244], "isthing": 1, "id": 2003, "name": "ape"},
    {"color": [85, 246, 52], "isthing": 1, "id": 2004, "name": "snake"},
    {"color": [215, 90, 90], "isthing": 1, "id": 2005, "name": "monkey"},
    {"color": [118, 163, 73], "isthing": 1, "id": 2006, "name": "hand"},
    {"color": [212, 212, 195], "isthing": 1, "id": 2007, "name": "rabbit"},
    {"color": [178, 187, 222], "isthing": 1, "id": 2008, "name": "duck"},
    {"color": [237, 53, 48], "isthing": 1, "id": 2009, "name": "fish"},
    {"color": [44, 20, 142], "isthing": 1, "id": 2010, "name": "turtle"},
    {"color": [134, 236, 45], "isthing": 1, "id": 2011, "name": "leopard"},
    {"color": [223, 1, 186], "isthing": 1, "id": 2012, "name": "fox"},
    {"color": [18, 36, 101], "isthing": 1, "id": 2013, "name": "deer"},
    {"color": [68, 72, 128], "isthing": 1, "id": 2014, "name": "owl"},
    {"color": [28, 233, 245], "isthing": 1, "id": 2015, "name": "tiger"},
    {"color": [62, 152, 39], "isthing": 1, "id": 2016, "name": "shark"},
    {"color": [8, 184, 26], "isthing": 1, "id": 2017, "name": "mouse"},
    {"color": [11, 2, 56], "isthing": 1, "id": 2018, "name": "frog"},
    {"color": [248, 186, 112], "isthing": 1, "id": 2019, "name": "eagle"},
    {"color": [39, 142, 155], "isthing": 1, "id": 2020, "name": "earless_seal"},
]


def _get_coco_ytvis_instances_meta():
    thing_ids = [k["id"] for k in COCO_YTVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_YTVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 101, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_YTVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


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
