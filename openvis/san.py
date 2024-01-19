import logging

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.clip_adapter import SideAdapter
from .modeling.clip_adapter.text_prompt import get_predefined_templates

from .modeling.video_maskformer import VideoMaskFormer
from .modeling.minvis import MinVIS

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class SAN(VideoMaskFormer):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(self, *, clip_adapter: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.clip_adapter = clip_adapter

    @classmethod
    def from_config(self, cfg):
        args_dict = VideoMaskFormer.from_config(cfg)

        # open-vocabulary
        args_dict["clip_adapter"] = SideAdapter(text_templates=get_predefined_templates("vild"))

        return args_dict

    def get_class_name_list(self, dataset_name):
        class_names = [c.strip() for c in MetadataCatalog.get(dataset_name).thing_classes]
        return class_names

    def forward(self, batched_inputs):
        dataset_name = batched_inputs[0]["dataset_name"]

        class_names = self.get_class_name_list(dataset_name)
        self.sem_seg_head.num_classes = len(class_names)

        ori_images = []
        for video in batched_inputs:
            for frame in video["image"]:
                ori_images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in ori_images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        ori_images = ImageList.from_tensors(ori_images, self.size_divisibility)

        clip_mg_feats, clip_bk_feats = self.clip_adapter.front_encode_image(ori_images.tensor)
        text_feats = self.clip_adapter.encode_text(class_names)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, extra_feats=clip_mg_feats)

        clip_feats = self.clip_adapter.post_encode_image(clip_bk_feats, outputs['class_attn_biases'].flatten(0, 1))
        outputs["pred_logits"] = einops.rearrange(self.clip_adapter.cal_sim_logits(text_feats, clip_feats), '(b t) q c -> b t q c', b=len(batched_inputs)).mean(dim=1)

        if self.training:
            if 'aux_outputs' in outputs:
                for idx, pred in enumerate(outputs["aux_outputs"]):
                    clip_feats = self.clip_adapter.post_encode_image(clip_bk_feats, pred['class_attn_biases'].flatten(0, 1))
                    outputs['aux_outputs'][idx]['pred_logits'] = einops.rearrange(self.clip_adapter.cal_sim_logits(text_feats, clip_feats), '(b t) q c -> b t q c', b=len(batched_inputs)).mean(dim=1)

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

            mask_cls_result, mask_pred_result = retry_if_cuda_oom(self.postprocess)(mask_cls_results[0], mask_pred_results[0], images.tensor.shape[-2:])

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(self.num_queries, len(class_names), mask_cls_result, mask_pred_result, image_size, height, width)


@META_ARCH_REGISTRY.register()
class SANOnline(MinVIS):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.adapter = SideAdapter(text_templates=get_predefined_templates("vild"))

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

        ori_images = ImageList.from_tensors(ori_images, self.size_divisibility)

        clip_mg_feats, clip_bk_feats = self.adapter.front_encode_image(ori_images.tensor)
        text_feats = self.adapter.encode_text(class_names)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, clip_mg_feats, self.window_size)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features, extra_feats=clip_mg_feats)

        clip_feats = self.adapter.post_encode_image(clip_bk_feats, outputs['class_attn_biases'].flatten(0, 1))
        outputs["pred_logits"] = einops.rearrange(self.adapter.cal_sim_logits(text_feats, clip_feats), '(b t) q c -> b t q c', b=len(batched_inputs))

        if self.training:
            if 'aux_outputs' in outputs:
                for idx, pred in enumerate(outputs["aux_outputs"]):
                    clip_feats = self.adapter.post_encode_image(clip_bk_feats, pred['class_attn_biases'].flatten(0, 1))
                    outputs['aux_outputs'][idx]['pred_logits'] = einops.rearrange(self.adapter.cal_sim_logits(text_feats, clip_feats), '(b t) q c -> b t q c', b=len(batched_inputs))

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

            return self.inference_video(mask_cls_result, mask_pred_result, image_size, height, width)

    def run_window_inference(self, images_tensor, clip_feats, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features, extra_feats=[x[start_idx:end_idx] for x in clip_feats])
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['class_attn_biases']
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['class_attn_biases'] = torch.cat([x['class_attn_biases'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embeds'] = torch.cat([x['pred_embeds'] for x in out_list], dim=2).detach()

        return outputs
