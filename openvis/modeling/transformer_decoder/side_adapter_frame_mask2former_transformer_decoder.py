from functools import partial

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable

from .position_encoding import PositionEmbeddingSine2D
from .video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder, MLP, TRANSFORMER_DECODER_REGISTRY


class ConvMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class SideAdapterFrameMultiScaleMaskedTransformerDecoder(VideoMultiScaleMaskedTransformerDecoder):

    @configurable
    def __init__(self, clip_heads, mask_classification, **kwargs):
        super().__init__(mask_classification=False, **kwargs)

        hidden_dim = kwargs["hidden_dim"]
        self.clip_heads = clip_heads

        # use 2D positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine2D(N_steps, normalize=True)

        # attention bias predict embeds
        self.attn_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # Attention Bias Branch
        self.attn_mlp = ConvMLP(hidden_dim, hidden_dim, hidden_dim * self.clip_heads, 3, affine_func=partial(nn.Conv2d, kernel_size=1))

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = super().from_config(cfg, in_channels, mask_classification)

        ret["clip_heads"] = cfg.MODEL.CLIP_ADAPTER.CLIP_NUM_HEADS

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        bt, c          = mask_features.shape[:-2]
        attn_features  = F.interpolate(mask_features, scale_factor=0.25, mode='bilinear', align_corners=False)
        h, w           = attn_features.shape[-2:]
        attn_features  = self.attn_mlp(attn_features)
        attn_features  = attn_features.reshape(bt, self.clip_heads, c, h, w)

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        class_attn_biases, outputs_mask, attn_mask = self.forward_prediction_heads(output, attn_features, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(class_attn_biases)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            class_attn_biases, outputs_mask, attn_mask = self.forward_prediction_heads(output, attn_features, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(class_attn_biases)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # expand BT to B, T
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) n q h w -> b t n q h w', t=t)

        pred_embeds = self.decoder_norm(output)
        pred_embeds = einops.rearrange(pred_embeds, 'q (b t) c -> b t q c', t=t)

        out = {
            'class_attn_biases': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'attn_feats': attn_features,
            'mask_feats': mask_features,
            'ms_feats': src,
            'ms_pos': pos,
            'size_list': size_list,
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask),
            'pred_embeds': pred_embeds,
        }

        return out

    def forward_prediction_heads(self, output, attn_features, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        attn_embed     = self.attn_embed(decoder_output)
        mask_embed     = self.mask_embed(decoder_output)

        attn_biases   = torch.einsum("bqc,bnchw->bnqhw", attn_embed, attn_features)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return attn_biases, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"class_attn_biases": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
