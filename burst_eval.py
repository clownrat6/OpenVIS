
from openvis.data.evals.bursteval import BURSTesval

gt_json_path = 'datasets/burst/annotations/val/all_classes.json'

# base, novel: 17.24, 3.72
_infer_json_path = 'work_dirs/openvoc_ytvis_2019/swin/brivis_SwinB_bs16_6000st_ViT-L-336/eval/inference/burst_results.json'

# base, novel: 13.73, 2.73
# _infer_json_path = 'work_dirs/openvoc_ytvis_2019/brivis_v1_R50_bs16_6000st/eval/inference/burst_results.json'

# base, novel: 8.71, 3.9
# _infer_json_path = 'work_dirs/openvoc_ytvis_2019/openvis_R50_bs16_6000st/eval/inference/burst_results.json'

# base, novel:
# _infer_json_path = 'work_dirs/openvoc_ytvis_2019/swin/openvis_swinL_bs16_6000st_ViT-L-336/eval/inference/burst_results.json'

burst_api = BURSTesval(gt_json_path, _infer_json_path)

res = burst_api.evaluate()