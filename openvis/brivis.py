# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import einops
import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from .san import SANOnline
from .modeling.resampler import TemporalInstanceResampler
from .modeling.minvis import batch_video_match_via_embeds
from .modeling.matcher import VideoHungarianMatcher
from .modeling.criterion import VideoSetTrackingCriterion
from .modeling.brownian_criterion import BrownianBridgeCriterion
from .utils.index import batch_index


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class BriVIS(SANOnline):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(self, max_iter_num, **kwargs):
        super().__init__(**kwargs)
        # training based on pretrained san
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)
        for p in self.clip_adapter.parameters():
            p.requires_grad_(False)
        # these two variables are used for monitoring and modifying training process.
        self.iter = 0
        self.max_iter_num = max_iter_num

        # `nheads` is corresponding to CLIP version.
        self.resampler = TemporalInstanceResampler(hidden_dim=256, feed_dim=2048, nheads=8, nlayers=6)
        self.brownian_criterion = BrownianBridgeCriterion(hidden_dim=256, proj_dim=256)

    @classmethod
    def from_config(cls, cfg):
        args_dict = super().from_config(cfg)

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

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetTrackingCriterion(
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        args_dict['criterion'] = criterion
        args_dict['max_iter_num'] = cfg.SOLVER.MAX_ITER
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
        if self.training:
            dataset_name = "ytvis_2019_train2coco"
        else:
            dataset_name = list(set(x["dataset_name"] for x in batched_inputs))
            dataset_name = dataset_name[0]

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

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, clip_bk_feats, clip_mg_feats, text_feats)
        else:
            self.backbone.eval()
            self.sem_seg_head.eval()
            with torch.no_grad():
                features = self.backbone(images.tensor)
                image_outputs = self.sem_seg_head(features, extra_feats=clip_mg_feats)
                del image_outputs['aux_outputs']
                torch.cuda.empty_cache()

            # pred_embeds: (1, bt, q, c) -> (b, t, q, c)
            image_outputs['pred_embeds'] = einops.rearrange(image_outputs['pred_embeds'][0], '(b t) q c -> b t q c', b=len(batched_inputs))
            # class_attn_biases: (1, bt, n, q, h, w) -> (b, t, n, q, h, w)
            frame_attn_biases = einops.rearrange(image_outputs['class_attn_biases'][0], '(b t) n q h w -> b t n q h w', b=len(batched_inputs))
            # pred_logits: (b, t, q, c)
            clip_feats = self.clip_adapter.post_encode_image(clip_bk_feats, frame_attn_biases.flatten(0, 1))
            image_outputs["pred_logits"] = einops.rearrange(self.clip_adapter.cal_sim_logits(text_feats, clip_feats), '(b t) q c -> b t q c', b=len(batched_inputs))
            # pred_masks: (1, q, bt, h, w) -> (b, q, t, h, w)
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'][0], 'q (b t) h w -> b q t h w', b=len(batched_inputs))

            indices, frame_embeds = batch_video_match_via_embeds(image_outputs['pred_embeds'])
            image_outputs = self.reset_image_output_order(image_outputs, indices)

            outputs = self.resampler(frame_embeds, image_outputs['mask_feats'], image_outputs['attn_feats'], self.clip_adapter, clip_bk_feats, text_feats)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            outputs['aux_outputs'].append(image_outputs.copy())
            if self.iter < self.max_iter_num // 2:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(outputs, targets, image_outputs=image_outputs)
            else:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(outputs, targets, image_outputs=None)
            self.iter += 1

            # bipartite matching-based loss
            losses, indices = self.criterion(outputs, targets, matcher_outputs=image_outputs)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            losses['bc_loss'], losses['htm_loss'] = self.brownian_criterion(outputs['pred_embeds'])
            return losses
        else:
            mask_cls_result, mask_pred_result = retry_if_cuda_oom(self.post_processing)(outputs, images.tensor.shape[-2:])

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        # flatten the t frames as an image with size of (th, w)
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
        outputs['pred_logits'] = (outputs['pred_logits'][:, 0, :, :] + outputs['pred_logits'][:, -1, :, :]) / 2
        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
            image_outputs['pred_logits'] = image_outputs['pred_logits'].mean(dim=1)
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> b q () (t h) w')
                outputs['aux_outputs'][i]['pred_logits'] = (outputs['aux_outputs'][i]['pred_logits'][:, 0, :, :] + outputs['aux_outputs'][i]['pred_logits'][:, -1, :, :]) / 2

        gt_instances = []
        for targets_per_video in targets:
            targets_per_video['masks'] = einops.rearrange(targets_per_video['masks'], 'q t h w -> q () (t h) w')
            gt_instances.append(targets_per_video)
        return image_outputs, outputs, gt_instances

    def reset_image_output_order(self, outputs, indices):
        b, t, q = indices.shape

        frame_logits = batch_index(outputs['pred_logits'].flatten(0, 1), indices.flatten(0, 1))
        outputs['pred_logits'] = einops.rearrange(frame_logits, '(b t) q c -> b t q c', b=b)

        frame_masks = batch_index(outputs['pred_masks'].transpose(2, 1).flatten(0, 1), indices.flatten(0, 1))
        outputs['pred_masks'] = einops.rearrange(frame_masks, '(b t) q h w -> b q t h w', b=b)

        return outputs

    def post_processing(self, outputs, image_hw):
        """
        average the class logits and append query ids
        """
        # b, t, q, c -> b, q, c
        outputs['pred_logits'] = outputs['pred_logits'].mean(dim=1)
        mask_cls_result = outputs["pred_logits"][0]
        mask_pred_result = outputs["pred_masks"][0]

        ph, pw = mask_pred_result.shape[-2:]
        ih, iw = image_hw
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

        return mask_cls_result, mask_pred_result

    def run_window_inference(self, images_tensor, clip_bk_feats, clip_mg_feats, text_feats):
        iters = len(images_tensor) // self.window_size
        if len(images_tensor) % self.window_size != 0:
            iters += 1

        overall_mask_feats = []
        overall_attn_feats = []
        overall_frame_embds = []
        overall_ms_src = []
        overall_ms_pos = []
        overall_clip_bk_feats = []

        for i in range(iters):
            s_idx = i * self.window_size
            e_idx   = (i+1) * self.window_size

            # sementer inference
            features = self.backbone(images_tensor[s_idx:e_idx])
            image_outputs = self.sem_seg_head(features, extra_feats=[x[s_idx:e_idx] for x in clip_mg_feats])

            del features['res2'], features['res3'], features['res4'], features['res5']
            del image_outputs['pred_masks'], image_outputs['class_attn_biases']
            del image_outputs['aux_outputs']

            overall_frame_embds.append(image_outputs['pred_embeds'])
            overall_mask_feats.append(image_outputs['mask_feats'])
            overall_attn_feats.append(image_outputs['attn_feats'])
            overall_ms_src.append(image_outputs['ms_feats'])
            overall_ms_pos.append(image_outputs['ms_pos'])

            overall_clip_bk_feats.append({k: v[:, s_idx:e_idx] if 'cls_token' in str(k) else v[s_idx:e_idx] for k, v in clip_bk_feats.items()})

            torch.cuda.empty_cache()

        frame_embeds = torch.cat(overall_frame_embds, dim=1)
        indices, frame_embeds = batch_video_match_via_embeds(frame_embeds)

        overall_frame_embeds = [frame_embeds[:, i * self.window_size:(i + 1) * self.window_size] for i in range(iters)]

        size_list = image_outputs['size_list']

        outputs = self.resampler(overall_ms_src, overall_ms_pos, size_list, overall_frame_embeds, overall_mask_feats, overall_attn_feats, self.clip_adapter, overall_clip_bk_feats, text_feats)

        del overall_ms_src, overall_ms_pos, overall_frame_embeds, overall_mask_feats, overall_attn_feats, overall_clip_bk_feats, text_feats
        torch.cuda.empty_cache()

        outputs['pred_logits'] = outputs['pred_logits'].detach()
        outputs['pred_masks'] = outputs['pred_masks'].detach().cpu().to(torch.float32)

        return outputs
