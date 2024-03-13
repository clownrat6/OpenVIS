import os
import json

from tqdm import tqdm

ref_json = json.load(open("datasets/coco/annotations/instances_val2017.json", "r"))
coco_categories = ref_json['categories']
n2n_coco = {x['id']: x['name'] for x in coco_categories}

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

COCO_TO_YTVIS_2019 = {1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18, 21:15, 22:32, 23:20, 24:30, 25:22, 35:33, 36:33, 41:5, 42:27, 43:40}
COCO_TO_YTVIS_2021 = {1:26, 2:23, 3:5, 4:23, 5:1, 7:36, 8:37, 9:4, 16:3, 17:6, 18:9, 19:19, 21:7, 22:12, 23:2, 24:40, 25:18, 34:14, 35:31, 36:31, 41:29, 42:33, 43:34}
COCO_TO_OVIS = {1:1, 2:21, 3:25, 4:22, 5:23, 6:25, 8:25, 9:24, 17:3, 18:4, 19:5, 20:6, 21:7, 22:8, 23:9, 24:10, 25:11}
YTVIS_2019_TO_COCO = {v: k for k, v in COCO_TO_YTVIS_2019.items()}
YTVIS_2021_TO_COCO = {v: k for k, v in COCO_TO_YTVIS_2021.items()}
OVIS_TO_COCO = {v: k for k, v in COCO_TO_OVIS.items()}

convert_list = [
    (YTVIS_2019_TO_COCO, 
        os.path.join(_root, "ytvis_2019/train.json"),
        os.path.join(_root, "ytvis_2019/ytvis_2019_train2coco.json"), "YTVIS 2019 to COCO:"),
    # (YTVIS_2019_TO_COCO, 
    #     os.path.join(_root, "ytvis_2019/valid.json"),
    #     os.path.join(_root, "ytvis_2019/ytvis2019_valid2coco.json"), "YTVIS 2019 to COCO val:"),
    # (YTVIS_2021_TO_COCO, 
    #     os.path.join(_root, "ytvis_2021/train/instances.json"),
    #     os.path.join(_root, "ytvis_2021/ytvis2021_train2coco.json"), "YTVIS 2021 to COCO:"),
    # (YTVIS_2021_TO_COCO, 
    #     os.path.join(_root, "ytvis_2021/valid/instances.json"),
    #     os.path.join(_root, "ytvis_2021/ytvis2021_valid2coco.json"), "YTVIS 2021 to COCO val:"),
    # (OVIS_TO_COCO, 
    #     os.path.join(_root, "ovis/instances_train2017.json"),
    #     os.path.join(_root, "ovis/ovis_train2coco.json"), "OVIS to COCO:"),
]

for convert_dict, src_path, out_path, msg in convert_list:
    if not os.path.exists(src_path):
        continue

    def merge(src_cats, convert_dict, n2n):
        src_ids = [x['id'] for x in src_cats]
        left_ids = list(set(src_ids) - set(convert_dict.keys()))
        left_names = [n2n[x] for x in left_ids]

        return left_names, left_ids

    src_cats = json.load(open(src_path, "r"))['categories']
    src_n2n = {x['id']: x['name'] for x in src_cats}

    left_names, left_ids = merge(src_cats, convert_dict, src_n2n)

    merge_categories = []
    [merge_categories.append(x) for x in coco_categories]
    [merge_categories.append({'id': 2000 + idx, 'name': x}) for idx, x in enumerate(left_names)]

    # import random
    # print(merge_categories)
    # print(len(merge_categories))
    # [print('{' + f'\"color\": [{random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)}], \"isthing\": 1, \"id\": {x["id"]}, \"name\": \"{x["name"]}\"' + '},') for x in merge_categories]
    # exit(0)
    merge_convert_dict = {}
    merge_convert_dict.update(convert_dict)
    merge_convert_dict.update({x: 2000 + idx for idx, x in enumerate(left_ids)})

    src_f = open(src_path, "r")
    out_f = open(out_path, "w")
    src_json = json.load(src_f)
    # print(src_json.keys())   dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

    out_json = {}
    for k, v in src_json.items():
        if k != 'annotations' and k != 'categories':
            out_json[k] = v

    out_json['categories'] = merge_categories

    converted_item_num = 0
    out_json['annotations'] = []
    for anno in tqdm(src_json['annotations']):

        anno['category_id'] = merge_convert_dict[anno['category_id']]

        out_json['annotations'].append(anno)
        converted_item_num += 1

    json.dump(out_json, out_f)
    print(msg, converted_item_num, "items converted.")
