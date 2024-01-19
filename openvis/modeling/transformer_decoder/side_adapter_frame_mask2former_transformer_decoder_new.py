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
    def __init__(
        self,
        in_channels,
        mask_classification,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
    ):
        super().__init__(
            in_channels=in_channels, 
            mask_classification=False,
            num_classes=None,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            num_frames=num_frames,
        )

        # use 2D positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine2D(N_steps, normalize=True)

        # Attention Bias Branch
        self.total_layers = 1
        self.total_heads  = 12
        self.resolution = (224 // 16, 224 // 16)
        
        # attention bias predict embeds
        self.attn_embed = MLP(hidden_dim, hidden_dim * 12 * 1, hidden_dim * 12 * 1, 3)

        self.attn_mlp = ConvMLP(hidden_dim, hidden_dim, hidden_dim, 3, affine_func=partial(nn.Conv2d, kernel_size=1))

    def forward(self, x, mask_feats, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        # attn_feats = self.attn_mlp(F.interpolate(mask_feats, self.resolution, mode='bilinear', align_corners=False))
        attn_feats = self.attn_mlp(F.interpolate(mask_feats, scale_factor=0.25, mode='bilinear', align_corners=False))

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
        class_attn_biases, outputs_mask, attn_mask = self.forward_prediction_heads(output, attn_feats, mask_feats, attn_mask_target_size=size_list[0])
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

            class_attn_biases, outputs_mask, attn_mask = self.forward_prediction_heads(output, attn_feats, mask_feats, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
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
            'attn_feats': attn_feats,
            'mask_feats': mask_feats,
            'ms_feats': src,
            'ms_pos': pos,
            'size_list': size_list,
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask),
            'pred_embeds': pred_embeds,
        }

        return out

    def forward_prediction_heads(self, output, attn_feats, mask_feats, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        attn_embed     = self.attn_embed(decoder_output)
        mask_embed     = self.mask_embed(decoder_output)

        bs, q, c = decoder_output.shape
        attn_embed = attn_embed.reshape(bs, q, self.total_heads, c)

        attn_biases   = torch.einsum("bqnc,bchw->bnqhw", attn_embed, attn_feats)

        outputs_mask  = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feats)
        b, q, _, _ = outputs_mask.shape

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
