import torch
import clip
from PIL import Image

import sys
sys.path.append('.')
from openvis.data.datasets.ytvis_coco import YTVIS_COCO_CATEGORIES, YTVIS_2019_TO_COCO
from openvis.data.datasets.ytvis import YTVIS_CATEGORIES_2019, YTVIS_CATEGORIES_2021
from openvis.data.datasets.burst import ALL_BURST_CATEGORIES, COMMON_BURST_CATEGORIES
from openvis.data.datasets.lvvis import LVVIS_CATEGORIES

COCO_INSTANCE_CATEGORIES = [
    {'name': 'person',         'id': 1,  'isthing': 1, 'color': [22, 129, 67]},
    {'name': 'bicycle',        'id': 2,  'isthing': 1, 'color': [77, 28, 63]},
    {'name': 'car',            'id': 3,  'isthing': 1, 'color': [36, 199, 223]},
    {'name': 'motorcycle',     'id': 4,  'isthing': 1, 'color': [223, 229, 119]},
    {'name': 'airplane',       'id': 5,  'isthing': 1, 'color': [168, 102, 19]},
    {'name': 'bus',            'id': 6,  'isthing': 1, 'color': [254, 151, 155]},
    {'name': 'train',          'id': 7,  'isthing': 1, 'color': [72, 73, 72]},
    {'name': 'truck',          'id': 8,  'isthing': 1, 'color': [88, 236, 45]},
    {'name': 'boat',           'id': 9,  'isthing': 1, 'color': [223, 13, 195]},
    {'name': 'traffic light',  'id': 10, 'isthing': 1, 'color': [119, 171, 185]},
    {'name': 'fire hydrant',   'id': 11, 'isthing': 1, 'color': [244, 93, 146]},
    {'name': 'stop sign',      'id': 13, 'isthing': 1, 'color': [190, 46, 9]},
    {'name': 'parking meter',  'id': 14, 'isthing': 1, 'color': [129, 156, 177]},
    {'name': 'bench',          'id': 15, 'isthing': 1, 'color': [40, 229, 98]},
    {'name': 'bird',           'id': 16, 'isthing': 1, 'color': [193, 76, 8]},
    {'name': 'cat',            'id': 17, 'isthing': 1, 'color': [122, 26, 123]},
    {'name': 'dog',            'id': 18, 'isthing': 1, 'color': [130, 165, 162]},
    {'name': 'horse',          'id': 19, 'isthing': 1, 'color': [174, 23, 102]},
    {'name': 'sheep',          'id': 20, 'isthing': 1, 'color': [143, 172, 204]},
    {'name': 'cow',            'id': 21, 'isthing': 1, 'color': [211, 177, 62]},
    {'name': 'elephant',       'id': 22, 'isthing': 1, 'color': [76, 250, 192]},
    {'name': 'bear',           'id': 23, 'isthing': 1, 'color': [249, 56, 97]},
    {'name': 'zebra',          'id': 24, 'isthing': 1, 'color': [214, 85, 94]},
    {'name': 'giraffe',        'id': 25, 'isthing': 1, 'color': [246, 39, 141]},
    {'name': 'backpack',       'id': 27, 'isthing': 1, 'color': [28, 101, 14]},
    {'name': 'umbrella',       'id': 28, 'isthing': 1, 'color': [43, 244, 44]},
    {'name': 'handbag',        'id': 31, 'isthing': 1, 'color': [181, 234, 114]},
    {'name': 'tie',            'id': 32, 'isthing': 1, 'color': [39, 20, 103]},
    {'name': 'suitcase',       'id': 33, 'isthing': 1, 'color': [93, 71, 105]},
    {'name': 'frisbee',        'id': 34, 'isthing': 1, 'color': [92, 172, 222]},
    {'name': 'skis',           'id': 35, 'isthing': 1, 'color': [23, 210, 217]},
    {'name': 'snowboard',      'id': 36, 'isthing': 1, 'color': [93, 18, 101]},
    {'name': 'sports ball',    'id': 37, 'isthing': 1, 'color': [205, 208, 232]},
    {'name': 'kite',           'id': 38, 'isthing': 1, 'color': [23, 48, 25]},
    {'name': 'baseball bat',   'id': 39, 'isthing': 1, 'color': [4, 161, 115]},
    {'name': 'baseball glove', 'id': 40, 'isthing': 1, 'color': [134, 194, 108]},
    {'name': 'skateboard',     'id': 41, 'isthing': 1, 'color': [200, 38, 223]},
    {'name': 'surfboard',      'id': 42, 'isthing': 1, 'color': [162, 172, 167]},
    {'name': 'tennis racket',  'id': 43, 'isthing': 1, 'color': [198, 150, 191]},
    {'name': 'bottle',         'id': 44, 'isthing': 1, 'color': [177, 54, 67]},
    {'name': 'wine glass',     'id': 46, 'isthing': 1, 'color': [90, 222, 17]},
    {'name': 'cup',            'id': 47, 'isthing': 1, 'color': [107, 146, 237]},
    {'name': 'fork',           'id': 48, 'isthing': 1, 'color': [147, 210, 203]},
    {'name': 'knife',          'id': 49, 'isthing': 1, 'color': [250, 116, 83]},
    {'name': 'spoon',          'id': 50, 'isthing': 1, 'color': [236, 166, 200]},
    {'name': 'bowl',           'id': 51, 'isthing': 1, 'color': [139, 92, 129]},
    {'name': 'banana',         'id': 52, 'isthing': 1, 'color': [194, 176, 128]},
    {'name': 'apple',          'id': 53, 'isthing': 1, 'color': [217, 137, 186]},
    {'name': 'sandwich',       'id': 54, 'isthing': 1, 'color': [199, 207, 32]},
    {'name': 'orange',         'id': 55, 'isthing': 1, 'color': [156, 96, 171]},
    {'name': 'broccoli',       'id': 56, 'isthing': 1, 'color': [133, 254, 99]},
    {'name': 'carrot',         'id': 57, 'isthing': 1, 'color': [203, 178, 251]},
    {'name': 'hot dog',        'id': 58, 'isthing': 1, 'color': [202, 97, 35]},
    {'name': 'pizza',          'id': 59, 'isthing': 1, 'color': [159, 219, 14]},
    {'name': 'donut',          'id': 60, 'isthing': 1, 'color': [113, 16, 234]},
    {'name': 'cake',           'id': 61, 'isthing': 1, 'color': [42, 118, 0]},
    {'name': 'chair',          'id': 62, 'isthing': 1, 'color': [133, 224, 232]},
    {'name': 'couch',          'id': 63, 'isthing': 1, 'color': [209, 210, 215]},
    {'name': 'potted plant',   'id': 64, 'isthing': 1, 'color': [34, 113, 60]},
    {'name': 'bed',            'id': 65, 'isthing': 1, 'color': [65, 4, 173]},
    {'name': 'dining table',   'id': 67, 'isthing': 1, 'color': [114, 243, 93]},
    {'name': 'toilet',         'id': 70, 'isthing': 1, 'color': [197, 110, 101]},
    {'name': 'tv',             'id': 72, 'isthing': 1, 'color': [30, 71, 91]},
    {'name': 'laptop',         'id': 73, 'isthing': 1, 'color': [22, 206, 242]},
    {'name': 'mouse',          'id': 74, 'isthing': 1, 'color': [89, 85, 68]},
    {'name': 'remote',         'id': 75, 'isthing': 1, 'color': [201, 128, 237]},
    {'name': 'keyboard',       'id': 76, 'isthing': 1, 'color': [214, 123, 154]},
    {'name': 'cell phone',     'id': 77, 'isthing': 1, 'color': [191, 174, 253]},
    {'name': 'microwave',      'id': 78, 'isthing': 1, 'color': [53, 45, 223]},
    {'name': 'oven',           'id': 79, 'isthing': 1, 'color': [149, 239, 254]},
    {'name': 'toaster',        'id': 80, 'isthing': 1, 'color': [89, 34, 19]},
    {'name': 'sink',           'id': 81, 'isthing': 1, 'color': [59, 96, 117]},
    {'name': 'refrigerator',   'id': 82, 'isthing': 1, 'color': [134, 58, 16]},
    {'name': 'book',           'id': 84, 'isthing': 1, 'color': [72, 100, 13]},
    {'name': 'clock',          'id': 85, 'isthing': 1, 'color': [126, 201, 38]},
    {'name': 'vase',           'id': 86, 'isthing': 1, 'color': [131, 90, 113]},
    {'name': 'scissors',       'id': 87, 'isthing': 1, 'color': [60, 83, 36]},
    {'name': 'teddy bear',     'id': 88, 'isthing': 1, 'color': [167, 128, 5]},
    {'name': 'hair drier',     'id': 89, 'isthing': 1, 'color': [18, 205, 90]},
    {'name': 'toothbrush',     'id': 90, 'isthing': 1, 'color': [162, 162, 173]},
]

# ytvis_category_ids = [x['id'] for x in YTVIS_CATEGORIES_2019]
# coco_category_ids = [x['id'] for x in COCO_INSTANCE_CATEGORIES]

# ytvis_coco_category_ids = [YTVIS_2019_TO_COCO[x['id']] for x in YTVIS_CATEGORIES_2019]

# res = set(ytvis_category_ids) - set(YTVIS_2019_TO_COCO.keys())

# 生成式开放词汇分类

# print(res, len(res))

# exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_texts(texts):
    clip_texts = clip.tokenize(texts).to(device)
    text_features = model.encode_text(clip_texts)

    return text_features / text_features.norm(dim=1, keepdim=True)

ytvis_coco_cats = [x['name'] for x in YTVIS_COCO_CATEGORIES] 
burst_cats = [(x['name'], x['lvis_id']) for x in ALL_BURST_CATEGORIES]
# common_burst_cats = [x['name'] for x in ALL_BURST_CATEGORIES if x['lvis_id'] in COMMON_BURST_CATEGORIES]
lvvis_cats = [(x['name'], x['id']) for x in LVVIS_CATEGORIES]

ytvis_coco_embeds = embed_texts(ytvis_coco_cats)

# base_cats = []
# base_ids = []
# novel_cats = []
# count = 0
# for cat, lvvis_id in burst_cats:
#     if cat in ytvis_coco_cats:
#         count += 1
#         base_cats.append(cat)
#         base_ids.append(lvvis_id)
#         continue
#     elif cat in ['glove', 'flowerpot', 'sword', 'toothpaste', 'bedspread', 'turkey_(bird)']:
#         novel_cats.append(cat)
#         continue
#     cat_embed = embed_texts([cat])
#     sim = (ytvis_coco_embeds @ cat_embed.T).squeeze(-1)
#     max_val, max_id = torch.topk(sim, k=1)
#     max_val = max_val.item()
#     max_id = max_id.item()
#     if max_val > 0.9:
#         count += 1
#         base_cats.append(cat)
#         base_ids.append(lvvis_id)
#         print(f'{cat} vs {ytvis_coco_cats[max_id]} - {max_val:.2f}')
#     else:
#         novel_cats.append(cat)

# print(len(base_cats))
# print(len(novel_cats))

# base_cats = []
# novel_cats = []

# print(count)

# print(base_ids)

# exit(0)

base_cats = []
base_ids = []
novel_cats = []
count = 0
for cat, lvvis_id in lvvis_cats:
    if cat in ytvis_coco_cats:
        count += 1
        base_cats.append(cat)
        base_ids.append(lvvis_id)
        continue
    elif cat in ['bait', 'bible', 'bread', 'car_jack', 'cell_phone_charger', 'desk', 'flowerpot', 'glove', 'roast_duck', 'sword', 'toothpaste', 'tricycle']:
        novel_cats.append(cat)
        continue
    cat_embed = embed_texts([cat])
    sim = (ytvis_coco_embeds @ cat_embed.T).squeeze(-1)
    max_val, max_id = torch.topk(sim, k=1)
    max_val = max_val.item()
    max_id = max_id.item()
    if max_val > 0.9:
        count += 1
        base_cats.append(cat)
        base_ids.append(lvvis_id)
        print(f'{cat} vs {ytvis_coco_cats[max_id]} - {max_val:.2f}')
    else:
        novel_cats.append(cat)
    # if cat in ytvis_coco_cats:
    #     print(f'{cat} - {sim:.2f}')

print(count)
print(base_ids)
print(novel_cats)