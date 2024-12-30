from tabulate import tabulate
from trackeval import Evaluator
from trackeval.datasets import BURST
from trackeval.metrics import TrackMAP, CLEAR, Identity, HOTA

from .burst import BURSTSingle

class BURSTesval:

    def __init__(self, gt_json, infer_json):
        # Command line interface:
        default_eval_config = Evaluator.get_default_eval_config()
        default_eval_config['PRINT_ONLY_COMBINED'] = True
        default_eval_config['DISPLAY_LESS_PROGRESS'] = True
        default_eval_config['PLOT_CURVES'] = False
        default_eval_config["OUTPUT_DETAILED"] = False
        default_eval_config["PRINT_RESULTS"] = False
        default_eval_config["OUTPUT_SUMMARY"] = False

        default_dataset_config = BURST.get_default_dataset_config()

        default_metrics_config = {'METRICS': ['HOTA', 'TrackMAP']}

        config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

        config['USE_PARALLEL'] = True
        config['EXEMPLAR_GUIDER'] = False
        config['GT_FOLDER'] = gt_json
        config['TRACKERS_FOLDER'] = infer_json

        self.eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        self.dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        self.metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    def _prepare(self):

        # Run code
        self.evaluator = Evaluator(self.eval_config)

        self.dataset_list = [BURSTSingle(self.dataset_config)]
        self.metrics_list = []
        for metric in [TrackMAP, CLEAR, Identity, HOTA]:
            if metric.get_name() in self.metrics_config['METRICS']:
                self.metrics_list.append(metric())

        if len(self.metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')

    def evaluate(self):
        self._prepare()
        output_res, output_msg = self.evaluator.evaluate(self.dataset_list, self.metrics_list, show_progressbar=True)

        trackers = list(output_res["BURSTSingle"].keys())
        assert len(trackers) == 1
        res = output_res['BURSTSingle'][trackers[0]]['COMBINED_SEQ']

        def average_metric(m):
            return round(100*sum(m) / len(m), 2)

        all_names = [x for x in res.keys() if (x != 'cls_comb_cls_av') and (x != 'cls_comb_det_av')]
        class_name_to_id = {x['name']: x['id'] for x in self.dataset_list[0].gt_data['categories']}
        known_list = [4, 13, 1038, 544, 1057, 34, 35, 36, 41, 45, 58, 60, 579, 1091, 1097, 1099, 78, 79, 81, 91, 1115,
                      1117, 95, 1122, 99, 1132, 621, 1135, 625, 118, 1144, 126, 642, 1155, 133, 1162, 139, 154, 174, 185,
                      699, 1215, 714, 717, 1229, 211, 729, 221, 229, 747, 235, 237, 779, 276, 805, 299, 829, 852, 347,
                      371, 382, 896, 392, 926, 937, 428, 429, 961, 452, 979, 980, 982, 475, 480, 993, 1001, 502, 1018]
        base_list = [4, 20, 34, 35, 36, 41, 45, 58, 60, 78, 79, 81, 91, 95, 99, 108, 118, 126, 127, 133, 139, 174, 176, 211, 221, 229, 235, 237, 276, 299, 347, 363, 365, 371, 382, 407, 408, 415, 428, 429, 453, 475, 480, 481, 499, 502, 544, 579, 621, 625, 642, 664, 672, 709, 714, 716, 780, 805, 813, 829, 844, 875, 877, 896, 926, 936, 937, 954, 960, 961, 980, 982, 1001, 1007, 1018, 1040, 1057, 1070, 1092, 1099, 1108, 1115, 1122, 1132, 1134, 1135, 1144, 1152, 1155, 1184, 1213, 1215, 1220, 1225, 1229]
        class_split_names = {
            "all": [x for x in all_names],
            "common": [x for x in all_names if class_name_to_id[x] in known_list],
            "uncommon": [x for x in all_names if class_name_to_id[x] not in known_list],
            "base": [x for x in all_names if class_name_to_id[x] in base_list],
            "novel": [x for x in all_names if class_name_to_id[x] not in base_list]
        }

        ret_results = {
            "all": {},
            "common": {},
            "uncommon": {},
            "base": {},
            "novel": {},
        }

        # metrics = ("AP", "AP50", "AP75", "AP_area_s", "AP_area_m", "AP_area_l", "AP_time_s", "AP_time_m", "AP_time_l", "HOTA", "DetA", "AssA")
        metrics = ("AP", "AP50", "AP75", "HOTA", "DetA", "AssA")
        for metric in metrics:
            for split_name in ["all", "common", "uncommon", "base", "novel"]:
                split_classes = class_split_names[split_name]

                if metric == "AP":
                    ret_results[split_name.lower()][metric] = average_metric([res[c]['TrackMAP']["AP_all"].mean() for c in split_classes])
                elif metric == "AP50":
                    # AP_all contains 10 AP metric values (AP50 -> AP95).
                    ret_results[split_name.lower()][metric] = average_metric([res[c]['TrackMAP']["AP_all"][0] for c in split_classes])
                elif metric == "AP75":
                    ret_results[split_name.lower()][metric] = average_metric([res[c]['TrackMAP']["AP_all"][5] for c in split_classes])
                elif metric in ["AP_area_s", "AP_area_m", "AP_area_l", "AP_time_s", "AP_time_m", "AP_time_l"]:
                    ret_results[split_name.lower()][metric] = average_metric([res[c]['TrackMAP'][metric].mean() for c in split_classes])
                else:
                    ret_results[split_name.lower()][metric] = average_metric([res[c]['HOTA'][metric].mean() for c in split_classes])

        # show row metrics
        table_data = []
        for metric in metrics:
            row = [metric]
            for split_name in ["all", "common", "uncommon", "base", "novel"]:
                row.append(ret_results[split_name][metric])
            table_data.append(row)

        print(tabulate(table_data, ["metric", "all", "common", "uncommon", "base", "novel"]), '\n')

        return ret_results
