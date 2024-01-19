import logging

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.projects.point_rend.point_features import point_sample

from .modeling.video_maskformer import VideoMaskFormer
from .modeling.matcher import batch_dice_loss, VideoHungarianMatcher
from .modeling.clip_adapter.masqclip_adapter import MasQCLIPAdapter
from .modeling.criterion import VideoSetCriterion

logger = logging.getLogger(__name__)


class LabelAssigner(nn.Module):
    
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        assert cost_dice != 0, "all costs cant be 0"
        
    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            tgt_mask = targets[b]["masks"].to(out_mask)

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Assign each prediction a label
            cost_dice = cost_dice.reshape(num_queries, -1)
            if cost_dice.shape[1] == 0:
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=cost_dice.device),
                    torch.tensor([], dtype=torch.int64, device=cost_dice.device)
                ))
            else:
                min_val, min_idx = cost_dice.min(dim=1)  # [num_queries,]
                valid_query = (min_val < 0.40)
                indices.append((
                    torch.arange(cost_dice.shape[0], dtype=torch.int64, device=min_val.device)[valid_query],
                    min_idx[valid_query].to(torch.int64)
                ))

        return indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Assigner " + self.__class__.__name__
        body = [
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


@META_ARCH_REGISTRY.register()
class MasQCLIP(VideoMaskFormer):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for name, para in self.backbone.named_parameters():
            para.requires_grad = False
        for name, para in self.sem_seg_head.named_parameters():
            para.requires_grad = False

        self.clip_adapter = MasQCLIPAdapter()

    @classmethod
    def from_config(self, cfg):
        args_dict = VideoMaskFormer.from_config(cfg)

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels"]

        criterion = VideoSetCriterion(
            args_dict["sem_seg_head"].num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        args_dict["criterion"] = criterion

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

        self.backbone.eval()
        self.sem_seg_head.eval()
        with torch.no_grad():
            features = self.backbone(images.tensor)
            base_outputs = self.sem_seg_head(features)
            base_outputs.pop('aux_outputs')

        # (1, q, (b t), h, w) -> (b, q, t, h, w)
        masks = einops.rearrange(base_outputs["pred_masks"][0], "q (b t) h w -> b q t h w", b=len(batched_inputs))

        logits = self.clip_adapter(ori_images.tensor, masks.transpose(1, 2).flatten(0, 1), class_names)
        logits = einops.rearrange(logits, "(b t) q c -> b t q c", b=len(batched_inputs)).mean(dim=1)

        if self.training:
            outputs = {"pred_logits": logits, "pred_masks": masks}

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
            mask_score = F.log_softmax(base_outputs["pred_logits"], dim=-1)
            mask_cls_result = F.log_softmax(logits, dim=-1)
            mask_cls_result = (mask_score[:, :, [0]] + mask_cls_result)[:, :, :-1]
            mask_cls_result = torch.exp(mask_cls_result)[0]
            mask_pred_result = masks[0]

            ph, pw = mask_pred_result.shape[-2:]
            ih, iw = images.tensor.shape[-2:]
            if ph != ih or pw != iw:
                # upsample masks
                mask_pred_result = F.interpolate(
                    mask_pred_result,
                    size=(ih, iw),
                    mode="bilinear",
                    align_corners=False,
                )

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(self.num_queries, len(class_names), mask_cls_result, mask_pred_result, image_size, height, width)
