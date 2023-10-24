# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified by Zesen Cheng from
# https://github.com/facebookresearch/ov-seg/blob/main/open_vocab_seg/modeling/clip_adapter/adapter.py

from typing import List

import torch
from torch import nn
from torchvision.ops import roi_align
from detectron2.structures import BitMasks

from .utils import build_mask_adapted_clip_model
from .text_prompt import PromptExtractor

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


class FrameAdaptedClipAdapter(nn.Module):

    def __init__(self, clip_model_name: str, text_templates: PromptExtractor, mask_prompt_depth: int, mask_prompt_fwd: bool):
        super().__init__()
        self.clip_model = build_mask_adapted_clip_model(clip_model_name, mask_prompt_depth)
        self.text_templates = text_templates
        self.text_templates.init_buffer(self.clip_model)
        self.text_feature_buffer = {}

        self.mask_fill = [255.0 * c for c in PIXEL_MEAN]
        self.mask_prompt_fwd = mask_prompt_fwd
        self.register_buffer("pixel_mean", torch.Tensor(PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0)
        self.register_buffer("pixel_std", torch.Tensor(PIXEL_STD).reshape(1, 3, 1, 1) * 255.0)

    def forward(self, frames: torch.Tensor, text: List[str], masks: torch.Tensor, normalize: bool = True):
        regions, region_masks, valid_flag = self._preprocess_image(frames, masks, normalize=normalize)
        if regions is None:
            return None, valid_flag
        if self.mask_prompt_fwd:
            image_features = self.get_image_features(regions, region_masks)
        else:
            image_features = self.get_image_features(regions)
        text_feature = self.get_text_features(text)  # k,feat_dim
        return self.get_sim_logits(text_feature, image_features), valid_flag

    def _preprocess_image(self, frames: torch.Tensor, masks: torch.Tensor, normalize: bool = True):
        """crop, mask and normalize the image

        Args:
            image ([type]): [T,C,H,W]
            mask ([type]): [T,N,H,W]
            normalize (bool, optional): [description]. Defaults to True.
        """
        bin_masks = masks > 0.5
        valid = bin_masks.sum(dim=(-1, -2)) > 0
        if torch.sum(valid) == 0:
            return None, None, valid
        valid_bin_masks = bin_masks[valid]
        valid_masks = masks[valid]
        valid_bin_masks = BitMasks(valid_bin_masks)
        valid_bboxes = valid_bin_masks.get_bounding_boxes()
        valid_bboxes = valid_bboxes.tensor.cuda().contiguous()

        ids = torch.nonzero(valid)

        ind_boxes = torch.cat([ids[:, 0:1], valid_bboxes], dim=-1)
        regions = roi_align(frames.half(), ind_boxes.half(), output_size=(224, 224))

        ind_boxes = torch.cat([torch.arange(len(valid_bboxes), device=frames.device)[:, None], valid_bboxes], dim=-1)
        mask_regions = roi_align(valid_masks.half()[:, None], ind_boxes.half(), output_size=(224, 224))

        regions = mask_regions * regions + (1 - mask_regions) * self.pixel_mean
        if normalize:
            regions = (regions - self.pixel_mean) / self.pixel_std

        return regions, mask_regions, valid

    def _get_text_features(self, noun_list: List[str]):
        left_noun_list = [noun for noun in noun_list if noun not in self.text_feature_buffer]
        if len(left_noun_list) > 0:
            left_text_features = self.text_templates(left_noun_list, self.clip_model)
            self.text_feature_buffer.update({noun: text_feature for noun, text_feature in zip(left_noun_list, left_text_features)})
        return torch.stack([self.text_feature_buffer[noun] for noun in noun_list])

    def get_text_features(self, noun_list: List[str]):
        text_features = self._get_text_features(noun_list)
        return self.normalize_feature(text_features)

    def get_image_features(self, image: torch.Tensor):
        image_features = self.clip_model.visual(image)
        return self.normalize_feature(image_features)

    def get_sim_logits(self, text_features: torch.Tensor, image_features: torch.Tensor, temperature: float = 100):
        return temperature * image_features @ text_features.T

    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)


class VideoQueryAdaptedClipAdapter(FrameAdaptedClipAdapter):

    def __init__(self, clip_model_name: str, text_templates: PromptExtractor, mask_prompt_depth: int = 3, mask_prompt_fwd: bool = False):
        super().__init__(clip_model_name, text_templates, mask_prompt_depth, mask_prompt_fwd)
        self.non_object_embedding = nn.Parameter(torch.empty(1, self.clip_model.text_projection.shape[-1]))
        nn.init.normal_(self.non_object_embedding.data, std=self.clip_model.transformer.width ** -0.5,)

    def get_text_features_w_noobj(self, noun_list: List[str]):
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (self.non_object_embedding / self.non_object_embedding.norm(dim=-1, keepdim=True))
        return torch.cat([object_text_features, non_object_text_features], dim=0)
