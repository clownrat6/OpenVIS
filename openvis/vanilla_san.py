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


@META_ARCH_REGISTRY.register()
class VanillaSANOnline(MinVIS):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(self, *, clip_adapter: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.clip_adapter = clip_adapter

        # for name, para in self.clip_adapter.named_parameters():
        #     para.requires_grad = False

    @classmethod
    def from_config(self, cfg):
        args_dict = MinVIS.from_config(cfg)
        # hardcode to 1
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1

        # open-vocabulary
        args_dict["clip_adapter"] = SideAdapter(text_templates=get_predefined_templates("vild"))

        return args_dict

    def get_class_name_list(self, dataset_name):
        class_names = [c.strip() for c in MetadataCatalog.get(dataset_name).thing_classes]
        return class_names

    def forward(self, batched_inputs):
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

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, self.window_size)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            # remove classes
            for i in range(len(targets)):
                targets[i]['labels'] = torch.zeros_like(targets[i]['labels'])
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
            ori_frames = ImageList.from_tensors(ori_images, self.size_divisibility).tensor
            outputs["pred_logits"] = self.clip_adapter(ori_frames, class_names, outputs['pred_masks'].transpose(1, 2).flatten(0, 1))
            outputs["pred_logits"] = einops.rearrange(outputs["pred_logits"], "(b t) q c -> b t q c", b=len(batched_inputs))

            outputs = self.post_processing(outputs)

            mask_cls_result = outputs["pred_logits"].mean(dim=1)[0]
            mask_pred_result = outputs["pred_masks"][0]

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

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return self.inference_video(mask_cls_result, mask_pred_result, image_size, height, width)
