# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList

from .video_maskformer import VideoMaskFormer
from ..utils.index import batch_index

logger = logging.getLogger(__name__)


def match_via_embeds(tgt_embeds, cur_embeds):
    cur_embeds = cur_embeds / cur_embeds.norm(dim=1)[:, None]
    tgt_embeds = tgt_embeds / tgt_embeds.norm(dim=1)[:, None]
    cos_sim = torch.mm(cur_embeds, tgt_embeds.transpose(0,1))

    cost_embd = 1 - cos_sim

    C = 1.0 * cost_embd
    C = C.cpu()

    indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
    indices = indices[1]  # permutation that makes current aligns to target

    return indices.tolist()


def batch_video_match_via_embeds(embeds):
    # pred_embeds: bs x t x q x c
    bs, t = embeds.shape[:2]
    batch_indices_list = []
    out_embeds = []
    for b in range(bs):
        last_frame_embeds = embeds[b, 0]
        indices_list = []
        embeds_list = []
        for i in range(t):
            indices = match_via_embeds(last_frame_embeds, embeds[b, i])
            last_frame_embeds = embeds[b, i][indices]

            indices_list.append(indices)
            embeds_list.append(last_frame_embeds)

        batch_indices_list.append(indices_list)
        out_embeds.append(torch.stack(embeds_list))

    # b, t, q, c
    out_embeds = torch.stack(out_embeds)
    # b, t, q
    batch_indices = torch.tensor(batch_indices_list, device=embeds.device)

    return batch_indices, out_embeds


@META_ARCH_REGISTRY.register()
class MinVIS(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
        window_size,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference
        self.window_size = window_size

    @classmethod
    def from_config(cls, cfg):
        args_dict = VideoMaskFormer.from_config(cfg)

        args_dict['window_inference'] = cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE
        args_dict['window_size'] = cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE

        return args_dict

    @property
    def device(self):
        return self.pixel_mean.device

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
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, self.window_size)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

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

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )

        gt_instances = []
        for targets_per_video in targets:
            # labels: N (num instances)
            # ids: N, num_labeled_frames
            # masks: N, num_labeled_frames, H, W
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})

        return outputs, gt_instances

    def match_from_embeds(self, tgt_embeds, cur_embeds):
        cur_embeds = cur_embeds / cur_embeds.norm(dim=1)[:, None]
        tgt_embeds = tgt_embeds / tgt_embeds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embeds, tgt_embeds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def post_processing_old(self, outputs):
        # pred_logits: bs x t x q x n
        # pred_masks:  bs x q x t x h x w
        # pred_embeds: bs x t x q x c
        pred_logits, pred_masks, pred_embeds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embeds']
        bs, t = pred_logits.shape[:2]
        pred_masks = einops.rearrange(pred_masks, 'b q t h w -> b t q h w')
        batch_indices_list = []
        out_logits = []
        out_masks = []
        for b in range(bs):
            frame_embeds = pred_embeds[b]
            last_frame_embeds = frame_embeds[0]
            indices_list = []
            logits_list = []
            masks_list = []
            for i in range(t):
                indices = self.match_from_embeds(last_frame_embeds, frame_embeds[i])
                last_frame_embeds = frame_embeds[i][indices]

                indices_list.append(indices)
                logits_list.append(pred_logits[b, i, indices])
                masks_list.append(pred_masks[b, i, indices])

            batch_indices_list.append(indices_list)
            out_logits.append(sum(logits_list) / len(logits_list))
            out_masks.append(torch.stack(masks_list))

        out_logits = torch.stack(out_logits)
        out_masks = torch.stack(out_masks).transpose(2, 1)
        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def post_processing(self, outputs):
        # pred_logits: bs x t x q x n
        # pred_masks:  bs x q x t x h x w
        # pred_embeds: bs x t x q x c
        pred_logits, pred_masks, pred_embeds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embeds']
        video_indices, _ = batch_video_match_via_embeds(pred_embeds)
        bs, t = pred_logits.shape[:2]
        pred_masks = einops.rearrange(pred_masks, 'b q t h w -> b t q h w')

        frame_logits = batch_index(pred_logits.flatten(0, 1), video_indices.flatten(0, 1))
        frame_masks = batch_index(pred_masks.flatten(0, 1), video_indices.flatten(0, 1))

        frame_logits = einops.rearrange(frame_logits, '(b t) q c -> b t q c', b=bs)
        frame_masks = einops.rearrange(frame_masks, '(b t) q h w -> b q t h w', b=bs)

        outputs['pred_logits'] = frame_logits
        outputs['pred_masks'] = frame_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach().cpu().to(torch.float32)
        outputs['pred_embeds'] = torch.cat([x['pred_embeds'] for x in out_list], dim=1).detach()

        return outputs

    def prepare_targets(self, targets, images):
        return VideoMaskFormer.prepare_targets(self.num_frames, targets, images)

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        return VideoMaskFormer.inference_video(self.num_queries, self.sem_seg_head.num_classes, pred_cls, pred_masks, img_size, output_height, output_width)
