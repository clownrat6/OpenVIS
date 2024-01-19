# Modified from https://github.com/haochenheheda/LVVIS
import copy
from typing import Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F

from scipy.optimize import linear_sum_assignment

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import ShapeSpec
from detectron2.layers.batch_norm import get_norm, FrozenBatchNorm2d

from .modeling.matcher import autocast, np, batch_dice_loss_jit, batch_sigmoid_ce_loss_jit, batch_sigmoid_ce_loss, batch_dice_loss
from .modeling.criterion import sigmoid_ce_loss_jit, dice_loss_jit, point_sample, get_uncertain_point_coords_with_randomness, calculate_uncertainty, is_dist_avail_and_initialized, get_world_size
from .modeling.clip_adapter import default_clip_adapter

from timm.models.helpers import build_model_with_cfg
from timm.models.resnet import ResNet, Bottleneck
from timm.models.resnet import default_cfgs as default_cfgs_resnet

from timm.models.resnet import ResNet, Bottleneck

model_params = {
    'resnet50_in21k': dict(block=Bottleneck, layers=[3, 4, 6, 3]),
}


def freeze_module(x):
    """
    """
    for p in x.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(x)
    return x


class CustomResNet(ResNet):
    def __init__(self, **kwargs):
        self.out_indices = kwargs.pop('out_indices')
        super().__init__(**kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        ret = [x]
        x = self.layer1(x)
        ret.append(x)
        x = self.layer2(x)
        ret.append(x)
        x = self.layer3(x)
        ret.append(x)
        x = self.layer4(x)
        ret.append(x)
        return [ret[i] for i in self.out_indices]


    def load_pretrained(self, cached_file):
        data = torch.load(cached_file, map_location='cpu')
        if 'state_dict' in data:
            self.load_state_dict(data['state_dict'])
        else:
            self.load_state_dict(data)

def create_timm_resnet(variant, out_indices, pretrained=False, **kwargs):
    params = model_params[variant]
    default_cfgs_resnet['resnet50_in21k'] = \
        copy.deepcopy(default_cfgs_resnet['resnet50'])
    default_cfgs_resnet['resnet50_in21k']['url'] = \
        'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'
    default_cfgs_resnet['resnet50_in21k']['num_classes'] = 11221

    return build_model_with_cfg(
        CustomResNet, variant, pretrained,
        default_cfg=default_cfgs_resnet[variant],
        out_indices=out_indices,
        pretrained_custom_load=True,
        **params,
        **kwargs)


class TIMM(Backbone):
    def __init__(self, base_name, out_levels, freeze_at=0, norm='FrozenBN', pretrained=False):
        super().__init__()
        out_indices = [x - 1 for x in out_levels]
        if base_name in model_params:
            self.base = create_timm_resnet(
                base_name, out_indices=out_indices, 
                pretrained=False)
        else:
            assert 0, base_name
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction']) \
            for i, f in enumerate(self.base.feature_info)] 
        self._out_features = ['layer{}'.format(x) for x in out_levels]
        self._out_feature_channels = {
            'layer{}'.format(l): feature_info[l - 1]['num_chs'] for l in out_levels}
        self._out_feature_strides = {
            'layer{}'.format(l): feature_info[l - 1]['reduction'] for l in out_levels}
        self._size_divisibility = max(self._out_feature_strides.values())
        if 'resnet' in base_name:
            self.freeze(freeze_at)
        if norm == 'FrozenBN':
            self = FrozenBatchNorm2d.convert_frozen_batchnorm(self)

        del self.base.global_pool
        del self.base.fc

    def freeze(self, freeze_at=0):
        """
        """
        if freeze_at >= 1:
            print('Frezing', self.base.conv1)
            self.base.conv1 = freeze_module(self.base.conv1)
        if freeze_at >= 2:
            print('Frezing', self.base.layer1)
            self.base.layer1 = freeze_module(self.base.layer1)

    def forward(self, x):
        features = self.base(x)
        ret = {k: v for k, v in zip(self._out_features, features)}
        return ret
    
    @property
    def size_divisibility(self):
        return self._size_divisibility


# @BACKBONE_REGISTRY.register()
# def build_timm_backbone(cfg, input_shape):
#     model = TIMM(
#         cfg.MODEL.TIMM.BASE_NAME, 
#         cfg.MODEL.TIMM.OUT_LEVELS,
#         freeze_at=cfg.MODEL.TIMM.FREEZE_AT,
#         norm=cfg.MODEL.TIMM.NORM,
#         pretrained=cfg.MODEL.TIMM.PRETRAINED,
#     )
#     return model


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_object: float = 1, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_object = cost_object
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_object != 0 or cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    def linear_sum_assignment_with_inf(self, cost_matrix):
        cost_matrix = np.asarray(cost_matrix)
        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")

        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            min_values = values.min()
            max_values = values.max()
            m = min(cost_matrix.shape)

            positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
            if max_inf:
                place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
            elif min_inf:
                place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

            cost_matrix[np.isinf(cost_matrix)] = place_holder
        return linear_sum_assignment(cost_matrix)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            #out_class_prob = outputs["pred_logits"][b][:,:-1].softmax(-1)  # [num_queries, num_classes]
            #out_object_prob = outputs["pred_object_logits"][b].softmax(-1)  # [num_queries, num_classes]
            #out_prob = torch.cat([out_class_prob * out_object_prob[:,0:1],out_object_prob[:,1:2]],dim=1)

            out_class_prob = outputs["pred_logits"][b][:,:-1].sigmoid()  # [num_queries, num_classes]
            out_object_prob = outputs["pred_object_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_prob = torch.cat([(out_class_prob * out_object_prob[:,0:1]) ** 0.5, out_object_prob[:,1:2]],dim=1)

            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]
            # gt masks are already padded when preparing target
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
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            #indices.append(linear_sum_assignment(C))
            indices.append(self.linear_sum_assignment_with_inf(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_object: {}".format(self.cost_object),
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class Criterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio,
                 importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        empty_weight[-1] = 0
        empty_object_weight = torch.ones(2)
        empty_object_weight[-1] = 0.4
        self.register_buffer("empty_weight", empty_weight)
        self.register_buffer("empty_object_weight", empty_object_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        B,Q = src_logits.shape[0], src_logits.shape[1]
        src_object_logits = outputs["pred_object_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_object_classes = (target_classes == self.num_classes).long()
        target_classes_binary = F.one_hot(target_classes, num_classes=self.num_classes+1)
        target_classes_binary = target_classes_binary[:,:,:self.num_classes].float()

        loss_ce = F.binary_cross_entropy_with_logits(src_logits[:, :, :-1], target_classes_binary, reduction='none') # B x C
        loss_ce = 1.7 * torch.sum(loss_ce * (1 - target_object_classes[:,:,None])) / (B * Q)

        loss_object_ce = F.cross_entropy(src_object_logits.transpose(1, 2), target_object_classes, self.empty_object_weight)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce, "loss_object_ce": loss_object_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class ZeroShotClassifier(nn.Module):
    def __init__(
        self,
        input_shape: ShapeSpec,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Sequential(nn.Linear(input_size, zs_weight_dim//2),
                nn.ReLU(),
                nn.Linear(zs_weight_dim//2, zs_weight_dim))

        self.frame_clip_adapter = default_clip_adapter()

    def forward(self, x, texts):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        zs_weight = self.frame_clip_adapter.get_text_features(texts)
        zs_weight = torch.cat([zs_weight, torch.zeros_like(zs_weight)[0:1]])
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=-1)
        logits = torch.einsum('bqc,nc->bqn', x, zs_weight)
        if self.use_bias:
            logits = logits + self.cls_bias
        return logits


@META_ARCH_REGISTRY.register()
class OV2Seg(nn.Module):
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
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # frames
        num_frames: int,
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

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.num_frames = num_frames
        self.classifier = ZeroShotClassifier(256, 512)

        self.window_inference = True
        self.window_size = 10

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        object_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_object=object_weight,
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_object_ce": object_weight, "loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = Criterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            'num_frames': cfg.INPUT.SAMPLING_FRAME_NUM,
        }

    @property
    def device(self):
        return self.pixel_mean.device

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
        dataset_name = list(set([x["dataset_name"] for x in batched_inputs]))
        assert len(dataset_name) == 1, "only supports single dataset."
        dataset_name = dataset_name[0]

        class_names = self.get_class_name_list(dataset_name)
        self.sem_seg_head.num_classes = len(class_names)

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, class_names)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

            outputs['pred_logits'] = self.classifier(outputs['pred_logits'], class_names)

        if self.training:
            if 'aux_outputs' in outputs:
                for idx, pred in enumerate(outputs["aux_outputs"]):
                    outputs['aux_outputs'][idx]['pred_logits'] = self.classifier(pred['pred_logits'], class_names)

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
            mask_object_cls_results = outputs["pred_obj_logits"]
            mask_per_frame_cls_results = outputs["pred_per_frame_logits"]
            mask_per_frame_object_cls_results = outputs["pred_per_frame_obj_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_object_cls_result = mask_object_cls_results[0]
            mask_pred_result = mask_pred_results[0]

            ph, pw = mask_pred_result.shape[-2:]
            ih, iw = images.tensor.shape[-2:]
            if mask_cls_result.shape[-1] == self.sem_seg_head.num_classes + 1:
                mask_cls_result = mask_cls_result[:, :-1].sigmoid()
                mask_object_cls_result = F.softmax(mask_object_cls_result, dim=-1)[:, :-1]
                mask_per_frame_cls_results = mask_per_frame_cls_results.sigmoid()[:, :, :-1]
                mask_per_frame_object_cls_results = mask_per_frame_object_cls_results.softmax(-1)[:, :, :-1]

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

            return retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_object_cls_result, mask_pred_result, mask_per_frame_cls_results, mask_per_frame_object_cls_results, image_size, height, width)

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        device = images.tensor.device
        gt_instances = []
        # prepare for each batch
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=device)
            gt_ids_per_video = []

            # collect gt_ids and gt_masks of each video
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)           # shape: [N, T]
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)                # shape: [N]

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]  # N
            gt_ids_per_video = gt_ids_per_video[valid_idx]                  # N, T
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()      # N, T, H, W

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video})

        return gt_instances

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q h w -> b q () h w')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(outputs['aux_outputs'][i]['pred_masks'], 'b q h w -> b q () h w')
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

    def instance_inference(self, mask_cls, mask_object_cls, mask_pred, mask_per_frame_cls, mask_per_frame_object_cls, img_size, output_height, output_width):
        scores = (mask_cls * mask_object_cls) ** 0.5

        mask_per_frame_object_scores = (mask_per_frame_cls * mask_per_frame_object_cls) ** 0.5

        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='trunc')
        mask_pred = mask_pred[topk_indices]

        mask_per_frame_object_scores = mask_per_frame_object_scores[topk_indices]
        Q = mask_per_frame_object_scores.shape[0]
        mask_per_frame_object_scores = mask_per_frame_object_scores[torch.arange(Q), :, labels_per_image]

        mask_ignore = mask_per_frame_object_scores < scores_per_image[:, None] * 0.1
        mask_pred[mask_ignore] = -1

        if mask_pred.device == torch.device('cpu'):
            mask_pred = mask_pred.float()

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * (mask_pred > 0).float().flatten(1)).sum(1) / ((mask_pred > 0).float().flatten(1).sum(1) + 1e-6)

        mask_pred = mask_pred[:, :, :img_size[0], :img_size[1]]
        mask_pred = F.interpolate(mask_pred, size=(output_height, output_width), mode="bilinear", align_corners=False)

        masks = mask_pred > 0.

        out_scores = (scores_per_image * mask_scores_per_image.to(scores_per_image.device)).tolist()
        out_labels = labels_per_image.tolist()
        out_masks = [m for m in masks.cpu()]

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def post_processing(self, outputs):
        # This function is only called when evaluation so the batch size in this context must be 1. 
        # pred_logits: bt x q x c
        # pred_obj_logits: bt x q x 2
        # pred_masks: bt x q x h x w
        # pred_embeds: bt x q x c
        pred_logits, pred_obj_logits, pred_masks, pred_embeds = outputs['pred_logits'], outputs['pred_object_logits'], outputs['pred_masks'], outputs['pred_embeds']

        pred_logits = list(torch.unbind(pred_logits))
        pred_obj_logits = list(torch.unbind(pred_obj_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embeds = list(torch.unbind(pred_embeds))

        out_logits = [pred_logits[0]]
        out_obj_logits = [pred_obj_logits[0]]
        out_masks = [pred_masks[0]]
        out_embeds = [pred_embeds[0]]

        for i in range(1, len(pred_logits)):
            indices = self.match_from_embds(out_embeds[-1], pred_embeds[i])

            out_logits.append(pred_logits[i][indices, :])
            out_obj_logits.append(pred_obj_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            alpha = 0.7
            tmp_pred_embds = alpha * pred_embeds[i][indices, :] + (1 - alpha) * out_embeds[-1]
            out_embeds.append(tmp_pred_embds)

        # shape: bt x q x c
        per_frame_out_logits = torch.stack(out_logits, dim=1)
        # shape: bt x q x 2
        per_frame_out_obj_logits = torch.stack(out_obj_logits, dim=1)

        # shape: q x c
        out_logits = sum(out_logits) / len(out_logits)
        # shape: q x 2
        out_obj_logits = sum(out_obj_logits) / len(out_obj_logits)
        # shape: q x t x h x w
        out_masks = torch.stack(out_masks, dim=1)

        # recover batch size dimension
        out_logits = out_logits.unsqueeze(0)
        out_obj_logits = out_obj_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_obj_logits'] = out_obj_logits
        outputs['pred_per_frame_logits'] = per_frame_out_logits
        outputs['pred_per_frame_obj_logits'] = per_frame_out_obj_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, class_names):
        iters = len(images_tensor) // self.window_size
        if len(images_tensor) % self.window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * self.window_size
            end_idx = (i+1) * self.window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            out['pred_logits'] = self.classifier(out['pred_logits'], class_names)

            del features['res2'], features['res3'], features['res4'], features['res5']
            out_list.append(out)

        # merge outputs
        outputs = {}

        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=0)
        outputs['pred_object_logits'] = torch.cat([x['pred_object_logits'] for x in out_list], dim=0)
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=0).detach().cpu().to(torch.float32)
        outputs['pred_embeds'] = torch.cat([x['pred_embeds'] for x in out_list], dim=0)

        return outputs
