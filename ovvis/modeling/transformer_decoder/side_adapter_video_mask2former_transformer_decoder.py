from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable

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
class SideAdapterVideoMultiScaleMaskedTransformerDecoder(VideoMultiScaleMaskedTransformerDecoder):

    @configurable
    def __init__(
        self,
        in_channels,
        *,
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
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        # descrepate the query-based classification class embedding
        self.mask_classification = False
        super().__init__(
            in_channels,
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
            # video related
            num_frames=num_frames)

        self.clip_dims = 768
        # attention bias predict embeds
        self.attn_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # Attention Bias Branch
        self.total_layers = 1
        self.total_heads  = 12
        self.attn_mlp = ConvMLP(
            hidden_dim,
            hidden_dim,
            hidden_dim * 12 * 1,
            3,
            affine_func=partial(nn.Conv2d, kernel_size=1),
        )

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels

        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM

        return ret

    def forward(self, ms_features, mask_features, mask=None):

        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        mask_features     = mask_features.view(bs, t, c_m, h_m, w_m)

        attn_features  = F.interpolate(mask_features.flatten(0, 1), scale_factor=0.25, mode='bilinear', align_corners=False)
        h, w           = attn_features.shape[-2:]
        attn_features  = self.attn_mlp(attn_features)
        attn_features  = attn_features.reshape(bs, t, self.total_heads, c_m, h, w)

        # x is a list of multi-scale feature
        assert len(ms_features) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(ms_features[i].shape[-2:])
            pos.append(self.pe_layer(ms_features[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
            src.append(self.input_proj[i](ms_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            _, c, hw = src[-1].shape
            pos[-1] = pos[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            src[-1] = src[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

        # mask_preds, attn_biases = self.side_adapter_network(images, clip_image_features)
        # class_embs = [self.clip_rec_head(clip_image_features, attn_bias, normalize=True) for attn_bias in attn_biases]  # [B,N,C]

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output      = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

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
            output = self.transformer_ffn_layers[i](output)

            class_attn_biases, outputs_mask, attn_mask = self.forward_prediction_heads(output, attn_features, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(class_attn_biases)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {'class_attn_biases': predictions_class[-1], 'pred_masks': predictions_mask[-1],
               'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)}
        return out

    def forward_prediction_heads(self, output, attn_features, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        attn_embed     = self.attn_embed(decoder_output)
        mask_embed     = self.mask_embed(decoder_output)

        # H, W = clip_features.shape[-2:]
        # attn_features = F.interpolate(mask_features, size=(H, W), mode="bilinear", align_corners=False) + clip_features
        # attn_features = self.attn_mlp(attn_features)

        attn_biases   = torch.einsum("bqc,btnchw->btnqhw", attn_embed, attn_features)

        outputs_mask  = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
        b, q, t, _, _ = outputs_mask.shape

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
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
