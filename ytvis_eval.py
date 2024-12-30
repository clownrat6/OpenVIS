import json

import matplotlib.pyplot as plt
from openvis.data.evals.ytvos import YTVOS
from openvis.data.evals.ytvis_eval import YTVOSeval

gt_json_file = 'datasets/lvvis/val_ytvis_style.json'
# gt_json_file = 'cvpr2024/temporal/lvvis_val_ss.json'
# gt_json_file = 'cvpr2024/temporal/lvvis_val_hs.json'

_infer_json_path = 'work_dirs/openvoc_ytvis_2019/brivis_v1_R50_bs16_6000st/eval/inference/ytvis_results.json'
# _infer_json_path = 'work_dirs/openvoc_ytvis_2019/swin/brivis_SwinB_bs16_6000st_ViT-L-336/eval/inference/ytvis_results.json'
# _infer_json_path = 'work_dirs/openvoc_ytvis_2019/openvis_R50_bs16_6000st/inference/results.json'
# _infer_json_path = 'work_dirs/openvoc_ytvis_2019/swin/openvis_swinL_bs16_6000st_ViT-L-336/inference/results.json'

coco_gt = YTVOS(gt_json_file)

coco_res = json.load(open(_infer_json_path, "r"))

# When evaluating mask AP, if the results contain bbox, cocoapi will
# use the box area as the area of the instance, instead of the mask area.
# This leads to a different definition of small/medium/large.
# We remove the bbox field to let mask AP use mask area.
for c in coco_res:
    c.pop("bbox", None)

coco_dt = coco_gt.loadRes(coco_res)
coco_eval = YTVOSeval(coco_gt, coco_dt)

# For COCO, the default max_dets_per_image is [1, 10, 100].
max_dets_per_image = [1, 10, 100]  # Default from COCOEval
coco_eval.params.maxDets = max_dets_per_image

_gts = coco_eval._gts
_dts = coco_eval._dts

# [6, 7, 17, 22, 38, 45, 51, 62, 64, 83, 84, 88, 96, 100, 107, 112, 123, 130, 135, 142, 153, 167, 176, 196, 207, 213, 219, 224, 261, 262, 274, 278, 282, 296, 314, 328, 332, 336, 346, 355, 371, 382, 401, 427, 442, 446, 448, 454, 455, 472, 474, 504, 517, 525, 550, 553, 594, 595, 598, 610, 632, 641, 677, 689, 694, 697, 707, 734, 735, 754, 773, 778, 780, 793, 807, 843, 860, 862, 878, 883, 884, 892, 896, 902, 903, 922, 926, 927, 948, 949, 950, 961, 963, 967, 970, 971, 974, 988, 1000, 1010, 1024, 1027, 1044, 1052, 1063, 1068, 1078, 1080, 1084, 1086, 1087, 1091, 1099, 1100, 1101, 1114, 1121, 1124, 1128, 1154, 1177, 1179, 1184, 1195]

base_cats = [6, 7, 17, 22, 38, 45, 51, 62, 64, 83, 84, 88, 96, 100, 107, 112, 123, 130, 135, 142, 153, 167, 176, 196, 207, 213, 219, 224, 261, 262, 274, 278, 282, 296, 314, 328, 332, 336, 346, 355, 371, 382, 401, 427, 442, 446, 448, 454, 455, 472, 474, 504, 517, 525, 550, 553, 594, 595, 598, 610, 632, 641, 677, 689, 694, 697, 707, 734, 735, 754, 773, 778, 780, 793, 807, 843, 860, 862, 878, 883, 884, 892, 896, 902, 903, 922, 926, 927, 948, 949, 950, 961, 963, 967, 970, 971, 974, 988, 1000, 1010, 1024, 1027, 1044, 1052, 1063, 1068, 1078, 1080, 1084, 1086, 1087, 1091, 1099, 1100, 1101, 1114, 1121, 1124, 1128, 1154, 1177, 1179, 1184, 1195]

coco_eval.params.catIds = [x for x in coco_eval.params.catIds if x in base_cats]

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
