import os
import json

from tqdm import tqdm

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

COCO_TO_YTVIS_2019 = {
    1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18, 21:15, 22:32, 23:20, 24:30, 25:22, 35:33, 36:33, 41:5, 42:27, 43:40
}
COCO_TO_YTVIS_2021 = {
    1:26, 2:23, 3:5, 4:23, 5:1, 7:36, 8:37, 9:4, 16:3, 17:6, 18:9, 19:19, 21:7, 22:12, 23:2, 24:40, 25:18, 34:14, 35:31, 36:31, 41:29, 42:33, 43:34
}

COCO_TO_OVIS = {
    1:1, 2:21, 3:25, 4:22, 5:23, 6:25, 8:25, 9:24, 17:3, 18:4, 19:5, 20:6, 21:7, 22:8, 23:9, 24:10, 25:11, 
}

convert_list = [
    (COCO_TO_YTVIS_2019, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/coco2ytvis2019_train.json"), "COCO to YTVIS 2019:"),
    (COCO_TO_YTVIS_2019, 
        os.path.join(_root, "coco/annotations/instances_val2017.json"),
        os.path.join(_root, "coco/coco2ytvis2019_val.json"), "COCO val to YTVIS 2019:"),
    (COCO_TO_YTVIS_2021, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/coco2ytvis2021_train.json"), "COCO to YTVIS 2021:"),
    (COCO_TO_YTVIS_2021, 
        os.path.join(_root, "coco/annotations/instances_val2017.json"),
        os.path.join(_root, "coco/coco2ytvis2021_val.json"), "COCO val to YTVIS 2021:"),
    (COCO_TO_OVIS, 
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/coco2ovis_train.json"), "COCO to OVIS:"),
]

for convert_dict, src_path, out_path, msg in convert_list:
    src_f = open(src_path, "r")
    out_f = open(out_path, "w")
    src_json = json.load(src_f)
    # print(src_json.keys())   dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

    out_json = {}
    for k, v in src_json.items():
        if k != 'annotations':
            out_json[k] = v

    converted_item_num = 0
    out_json['annotations'] = []
    for anno in tqdm(src_json['annotations']):
        if anno["category_id"] not in convert_dict:
            continue

        out_json['annotations'].append(anno)
        converted_item_num += 1

    json.dump(out_json, out_f)
    print(msg, converted_item_num, "items converted.")
