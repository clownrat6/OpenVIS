import logging

import einops
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList

from .modeling.video_maskformer import VideoMaskFormer
from .modeling.minvis import MinVIS
from .modeling.clip_adapter import build_clip_adapter

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class SimpleBaseline(VideoMaskFormer):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(self, *, clip_adapter: nn.Module, clip_ensemble: bool, clip_ensemble_weight: float, **kwargs):
        super().__init__(**kwargs)
        self.clip_adapter = clip_adapter
        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

    @classmethod
    def from_config(self, cfg):
        args_dict = VideoMaskFormer.from_config(cfg)

        clip_adapter = build_clip_adapter(cfg.MODEL.CLIP_ADAPTER)
        # open-vocabulary
        args_dict["clip_adapter"] = clip_adapter
        args_dict["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        args_dict["clip_ensemble_weight"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        return args_dict

    def get_class_name_list(self, dataset_name):
        class_names = [c.strip() for c in MetadataCatalog.get(dataset_name).thing_classes]
        return class_names

    def forward(self, batched_inputs):
        dataset_name = batched_inputs[0]["dataset_name"]

        class_names = self.get_class_name_list(dataset_name)
        self.sem_seg_head.num_classes = len(class_names)

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        feats = self.backbone(images.tensor)
        outputs = self.sem_seg_head(feats)
        text_feats = self.clip_adapter.encode_text(class_names)
        outputs["pred_logits"] = self.clip_adapter.cal_sim_logits(text_feats, self.clip_adapter.normalize(outputs["pred_logits"]))

        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i]["pred_logits"] = self.clip_adapter.cal_sim_logits(
                        text_feats,
                        self.clip_adapter.normalize(outputs["aux_outputs"][i]["pred_logits"]))

            # mask classification target
            targets = self.prepare_targets(self.num_frames, batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            ph, pw = mask_pred_result.shape[-2:]
            ih, iw = images.tensor.shape[-2:]
            if mask_cls_result.shape[-1] == self.sem_seg_head.num_classes + 1:
                mask_cls_result = F.softmax(mask_cls_result, dim=-1)[:, :-1]
            if ph != ih or pw != iw:
                # upsample masks
                mask_pred_result = F.interpolate(
                    mask_pred_result,
                    size=(ih, iw),
                    mode="bilinear",
                    align_corners=False,
                )

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            mask_cls_result, mask_pred_result = self.open_vocabulary_ensemble(mask_cls_result, mask_pred_result, torch.stack([x.to(self.device) for x in batched_inputs[0]["image"]]), class_names)

            return self.inference_video(self.num_queries, len(class_names), mask_cls_result, mask_pred_result, image_size, height, width)

    def open_vocabulary_ensemble(self, scores: torch.Tensor, masks: torch.Tensor, frames: torch.Tensor, class_names):
        if len(scores) > 0:
            if self.clip_ensemble:
                frame_len = len(frames)
                part_len = 10
                clip_cls = []
                valid_flag = []
                for idx in range(0, frame_len, part_len):
                    part_frames = frames[idx:idx+part_len]
                    part_masks = masks[:, idx:idx+part_len].sigmoid().transpose(0, 1).contiguous()
                    part_clip_cls, part_valid_flag = self.clip_adapter(part_frames, class_names, part_masks)
                    if part_clip_cls is None:
                        part_clip_cls = torch.empty(0, len(class_names) + 1, device=self.device)
                    clip_cls.append(part_clip_cls); valid_flag.append(part_valid_flag)
                # remove non-object logits
                clip_cls = torch.cat(clip_cls)[..., :-1].softmax(dim=-1)
                valid_flag = torch.cat(valid_flag)

                if torch.sum(valid_flag) == 0:
                    return scores, masks

                # M x 2 (frame_idx, query_idx)
                valid_ids = torch.nonzero(valid_flag)
                # N
                valid_query_flag = torch.sum(valid_flag, dim=0) > 0
                # N x 1 -> N
                valid_query_ids = torch.nonzero(valid_query_flag)[:, 0]

                # frame-level average
                query_clip_cls = [torch.mean(clip_cls[valid_ids[:, 1] == query_id], dim=0) for query_id in valid_query_ids]
                clip_cls = torch.stack(query_clip_cls)

                if self.clip_ensemble_weight > 0:
                    map_back_clip_cls = scores.new_ones(scores.shape)
                    map_back_clip_cls[valid_query_flag] = clip_cls
                    scores = torch.pow(scores, 1 - self.clip_ensemble_weight) * torch.pow(
                        map_back_clip_cls, self.clip_ensemble_weight)
                else:
                    scores = clip_cls
                    masks = masks[valid_query_flag]

        return scores, masks


@META_ARCH_REGISTRY.register()
class SimpleBaselineOnline(MinVIS):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(self, *, clip_adapter: nn.Module, clip_ensemble: bool, clip_ensemble_weight: float, **kwargs):
        super().__init__(**kwargs)
        self.clip_adapter = clip_adapter
        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

    @classmethod
    def from_config(self, cfg):
        args_dict = MinVIS.from_config(cfg)

        clip_adapter = build_clip_adapter(cfg.MODEL.CLIP_ADAPTER)
        # open-vocabulary
        args_dict["clip_adapter"]         = clip_adapter
        args_dict["clip_ensemble"]        = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        args_dict["clip_ensemble_weight"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        return args_dict

    def get_class_name_list(self, dataset_name):
        class_names = [c.strip() for c in MetadataCatalog.get(dataset_name).thing_classes]
        return class_names

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        dataset_name = batched_inputs[0]["dataset_name"]

        class_names = self.get_class_name_list(dataset_name)
        self.sem_seg_head.num_classes = len(class_names)

        ori_images = []
        for video in batched_inputs:
            for frame in video["image"]:
                ori_images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in ori_images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, self.window_size)
        else:
            feats = self.backbone(images.tensor)
            outputs = self.sem_seg_head(feats)

        text_feats = self.clip_adapter.encode_text(class_names)
        outputs["pred_logits"] = self.clip_adapter.cal_sim_logits(text_feats, self.clip_adapter.normalize(outputs["pred_logits"]))

        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i]["pred_logits"] = self.clip_adapter.cal_sim_logits(
                        text_feats,
                        self.clip_adapter.normalize(outputs["aux_outputs"][i]["pred_logits"]))

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs = self.post_processing(outputs)

            mask_cls_results = outputs["pred_logits"].mean(dim=1)
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            ph, pw = mask_pred_result.shape[-2:]
            ih, iw = images.tensor.shape[-2:]
            if mask_cls_result.shape[-1] == self.sem_seg_head.num_classes + 1:
                mask_cls_result = F.softmax(mask_cls_result, dim=-1)[:, :-1]
            if ph != ih or pw != iw:
                # upsample masks
                mask_pred_result = F.interpolate(
                    mask_pred_result,
                    size=(ih, iw),
                    mode="bilinear",
                    align_corners=False,
                )

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            mask_cls_result, mask_pred_result = self.open_vocabulary_ensemble(mask_cls_result, mask_pred_result, torch.stack([x.to(self.device) for x in batched_inputs[0]["image"]]), class_names)

            return self.inference_video(mask_cls_result, mask_pred_result, image_size, height, width)

    def open_vocabulary_ensemble(self, scores: torch.Tensor, masks: torch.Tensor, frames: torch.Tensor, class_names):
        if len(scores) > 0:
            if self.clip_ensemble:
                frame_len = len(frames)
                part_len = 10
                clip_cls = []
                valid_flag = []
                for idx in range(0, frame_len, part_len):
                    part_frames = frames[idx:idx+part_len]
                    part_masks = masks[:, idx:idx+part_len].sigmoid().transpose(0, 1).contiguous()
                    part_clip_cls, part_valid_flag = self.clip_adapter(part_frames, class_names, part_masks)
                    if part_clip_cls is None:
                        part_clip_cls = torch.empty(0, len(class_names) + 1, device=self.device)
                    clip_cls.append(part_clip_cls); valid_flag.append(part_valid_flag)
                # remove non-object logits
                clip_cls = torch.cat(clip_cls)[..., :-1]
                clip_cls = clip_cls.softmax(dim=-1)
                valid_flag = torch.cat(valid_flag)

                if torch.sum(valid_flag) == 0:
                    return scores, masks

                # M x 2 (frame_idx, query_idx)
                valid_ids = torch.nonzero(valid_flag)
                # N
                valid_query_flag = torch.sum(valid_flag, dim=0) > 0
                # N x 1 -> N
                valid_query_ids = torch.nonzero(valid_query_flag)[:, 0]

                # frame-level average
                query_clip_cls = [torch.mean(clip_cls[valid_ids[:, 1] == query_id], dim=0) for query_id in valid_query_ids]
                clip_cls = torch.stack(query_clip_cls)

                if self.clip_ensemble_weight > 0:
                    map_back_clip_cls = scores.new_ones(scores.shape)
                    map_back_clip_cls[valid_query_flag] = clip_cls
                    scores = torch.pow(scores, 1 - self.clip_ensemble_weight) * torch.pow(
                        map_back_clip_cls, self.clip_ensemble_weight)
                else:
                    scores = clip_cls
                    masks = masks[valid_query_flag]

        return scores, masks
