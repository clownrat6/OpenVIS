from typing import  List

import open_clip
import torch
from torch import nn
from torch.nn import functional as F
from open_clip import CLIP, tokenizer

from ...utils.index import batch_index

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


def cluster_dpc_knn(tokens, cluster_num, k=5, token_mask=None, index_sort=True):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = tokens
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, parent_ids = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, cluster_ids = torch.topk(score, k=cluster_num, dim=-1)
        if index_sort:
            cluster_ids, _ = torch.sort(cluster_ids, dim=-1)

        # extract related cluster-token socres
        dist_matrix = batch_index(dist_matrix, cluster_ids)

        assign_ids = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        batch_ids = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        tmp_ids = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        assign_ids[batch_ids.reshape(-1), cluster_ids.reshape(-1)] = tmp_ids.reshape(-1)

    return cluster_ids, assign_ids


def merge_tokens(tokens, assign_ids, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = tokens

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    batch_indices = torch.arange(B, device=x.device)[:, None]
    idx = assign_ids + batch_indices * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    return x_merged


def freeze_params(model, frozen_exclude=[]):
    if "all" in frozen_exclude:
        return
    for name, param in model.named_parameters():
        if not any([exclude in name for exclude in frozen_exclude]):
            param.requires_grad = False

    return model


def downsample2d(src, target_shape, method="nearest"):
    # src: [N,C,H,W]
    # target_shape: [H',W']
    # return: [N,C,H',W']
    if method in ["bicubic", "bilinear", "nearest"]:
        src = F.interpolate(src, size=target_shape, mode=method, align_corners=False)
    elif method == "avg":
        src = F.adaptive_avg_pool2d(src, output_size=target_shape)
    elif method == "max":
        src = F.adaptive_max_pool2d(src, output_size=target_shape)
    return src


def resize_pos_embed2d(
    posemb,
    src_shape,
    tgt_shape,
    num_prefix_tokens=1,
    interpolation="bicubic",
    antialias=False,
):
    """interpolate positional embedding from src_shape to tgt_shape. posemb: [N,L,C]"""
    if src_shape == tgt_shape:
        return posemb
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = (posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:])
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb

    posemb_grid = posemb_grid.permute(0, 2, 1).reshape(1, -1, src_shape[0], src_shape[1])

    posemb_grid = F.interpolate(
        posemb_grid,
        size=tgt_shape,
        mode=interpolation,
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, tgt_shape[0] * tgt_shape[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


class ClipOutput(dict):

    def __init__(self, spacial_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacial_shape = spacial_shape

    def save(self, idx: int, clip_feat: torch.Tensor):
        l, b, c = clip_feat.shape
        self[idx] = (clip_feat[1:].permute(1, 2, 0).reshape(b, c, *self.spacial_shape))  # n, c, h, w
        self[f"{idx}_cls_token"] = clip_feat[0:1]  # n, 1, c


class SideAdapter(nn.Module):

    def __init__(self, clip_model_name="ViT-B/16", embed_dims=256, num_queries=100, text_templates: List[str] = ["a photo of {}"]):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained="openai")
        self.clip: CLIP = model
        freeze_params(self.clip)
        self.broken_idx  = 9
        self.merge_ids   = [3, 6, 9]

        self.clip_vis_width = 768
        self.clip_out_dims = 512
        self.embed_dims    = embed_dims
        self.sos_token_num = num_queries

        self.attn_projs = nn.ModuleList()
        if self.embed_dims is not None:
            for _ in range(len(self.merge_ids)):
                self.attn_projs.append(nn.Conv2d(self.clip_vis_width, self.embed_dims, kernel_size=1))
                nn.init.kaiming_uniform_(self.attn_projs[-1].weight, a=1)
                nn.init.constant_(self.attn_projs[-1].bias, 0)
        else:
            [self.attn_projs.append(nn.Identity()) for _ in range(len(self.merge_ids))]

        self.templates = text_templates
        self.bg_embed = nn.Parameter(torch.randn(1, self.clip_out_dims))
        nn.init.normal_(self.bg_embed, std=self.bg_embed.shape[1] ** -0.5)

        self.text_cache = {}

    def front_vis_forward(self, frames):
        # scale_factor        = 0.5
        # if scale_factor    != 1.0:
        #     clip_frames     = F.interpolate(frames, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        clip_frames = F.interpolate(frames, (224, 224), mode="bilinear", align_corners=False)
        frame_len = len(frames)
        part_len = 10
        keys = list(set(self.merge_ids + [self.broken_idx, f'{self.broken_idx}_cls_token']))
        clip_video_features = {k: [] for k in keys}
        for idx in range(0, frame_len, part_len):
            part_frames = clip_frames[idx:idx+part_len]
            clip_frame_features = self.front_encode_image(part_frames)
            [clip_video_features[k].append(clip_frame_features[k]) for k in keys]

        clip_video_features = {k: torch.cat(clip_video_features[k], dim=1) if isinstance(k, str) and 'cls_token' in k else torch.cat(clip_video_features[k]) for k in keys}
        clip_mg_features = [v for k, v in clip_video_features.items() if k in self.merge_ids]
        clip_mg_features = [self.attn_projs[i](v) for i, v in enumerate(clip_mg_features)]
        clip_bk_features = {k: clip_video_features[k] for k in [self.broken_idx, f'{self.broken_idx}_cls_token']}

        return clip_mg_features, clip_bk_features

    def post_vis_forward(self, feats, attn_biases):
        return self.post_encode_image(feats, attn_biases, normalize=True)

    def get_sim_logits(self, feats, attn_biases, text_feats):
        bs, t = attn_biases.shape[:2]
        # B x T -> BT
        attn_biases = attn_biases.flatten(0, 1)
        class_feats = self.post_vis_forward(feats, attn_biases)
        bt, num_queries, _ = class_feats.shape

        # BT x Q x C -> B x T x Q x C
        class_feats = class_feats.reshape(bs, t, num_queries, -1)

        # (b, t, q, c) -> (b, q, t, c) -> (bq, t, c)
        # output = class_feats.transpose(2, 1).flatten(0, 1)
        # frag_ids, assign_ids = cluster_dpc_knn(output, 5)
        # frag_tokens = merge_tokens(output, assign_ids, 5, None)
        # frag_tokens = frag_tokens.transpose(0, 1)

        # frag_tokens = batch_index(output, frag_ids)

        # pseudo_class_feats = torch.zeros_like(class_feats).transpose(2, 1).flatten(0, 1)
        # pseudo_class_feats = batch_scatter(pseudo_class_feats, frag_ids, frag_tokens)
        # pseudo_class_feats = pseudo_class_feats.reshape(bs, num_queries, t, -1).transpose(2, 1)

        logits = torch.einsum("btqc,nc->btqn", class_feats, text_feats)
        # logits = torch.einsum("btqc,nc->btqn", pseudo_class_feats, text_feats)

        return logits

    def get_sim_logits_frame(self, feats, attn_biases, text_feats):
        class_feats = self.post_vis_forward(feats, attn_biases)

        logits = torch.einsum("bqc,nc->bqn", class_feats, text_feats)

        return logits

    def txt_forward(self, texts):
        return (self.clip.logit_scale.exp() * self.encode_text(texts))  # C+1,ndim

    def front_encode_image(self, x: torch.Tensor):
        cm = self.clip
        vis = cm.visual
        if vis.input_patchnorm:
            raise NotImplementedError("input_patchnorm is not implemented yet.")
        else:
            x = vis.conv1(x)  # shape = [*, width, grid, grid]
            _, _, h, w = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                # 768 -> 1x1x768 -> bsx1x768
                vis.class_embedding.to(x.dtype).reshape(1, 1, -1).repeat(x.shape[0], 1, 1),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        pos_embed = vis.positional_embedding.to(x.dtype)
        pos_embed = resize_pos_embed2d(pos_embed[None, ...], vis.grid_size, (h, w))[0]
        x = x + pos_embed

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = vis.patch_dropout(x)
        x = vis.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        resblocks = vis.transformer.resblocks[:self.broken_idx]
        outputs = ClipOutput(spacial_shape=(h, w))
        outputs.save(0, x)
        for i, resblock in enumerate(resblocks, start=1):
            x = resblock(x)
            outputs.save(i, x)
        return outputs

    def post_encode_image(self, feats, attn_bias, normalize=True):
        cm = self.clip
        # construct clip shadow features.
        cls_token = feats[f"{self.broken_idx}_cls_token"]  # n,1,c
        pix_feat = feats[self.broken_idx]  # n,c,h,w
        n, c, h, w = pix_feat.shape
        x = torch.cat([cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1)])  # 1+l,n,c

        # construct sos token.
        sos_token = cls_token.repeat(self.sos_token_num, 1, 1)

        if not isinstance(attn_bias, (list, tuple)):
            attn_bias = [attn_bias]

        resblocks = cm.visual.transformer.resblocks[self.broken_idx:]
        # construct attn biases.
        attn_biases = self._build_attn_biases(attn_bias,  resblocks[0].attn.num_heads, len(resblocks), target_shape=(h, w))
        x = torch.cat([sos_token, x], dim=0)
        for i, resblock in enumerate(resblocks):
            x = resblock(x, attn_mask=attn_biases[i])
        sos_token = x[: self.sos_token_num]

        sos_token = sos_token.permute(1, 0, 2)  # LND -> NLD

        sos_token = cm.visual.ln_post(sos_token)
        sos_token = sos_token @ cm.visual.proj
        if normalize:
            sos_token = F.normalize(sos_token, dim=-1)
        return sos_token

    def _build_attn_biases(self, attn_biases, num_heads, num_layers, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # convert it to proper format: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sos,H,W]
            n, num_head, num_sos, h, w = attn_bias.shape
            # reshape and downsample
            attn_bias = downsample2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                target_shape,
                method='max',
            )
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)
            true_num_head = num_heads
            assert (
                num_head == 1 or num_head == true_num_head
            ), f"num_head={num_head} is not supported."
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]
            # [n*num_head, num_sos+1+L, num_sos+1+L]
            new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L, num_sos + 1 + L)
            new_attn_bias[:, :num_sos] = -100
            new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0
            new_attn_bias[:num_sos, num_sos] = -100
            new_attn_bias = (new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone())
            new_attn_bias[..., :num_sos, -L:] = attn_bias
            formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [formatted_attn_biases[0] for _ in range(num_layers)]
        return formatted_attn_biases

    def encode_text(self, x: List[str], w_bg=True):
        cm = self.clip

        new_words = [word for word in x if word not in self.text_cache]
        if len(new_words) > 0:
            text_embeds_bucket = []
            for template in self.templates:
                noun_tokens = tokenizer.tokenize([template.format(noun) for noun in new_words])
                text_inputs = noun_tokens.to(self.bg_embed.device)
                text_embeds = cm.encode_text(text_inputs, normalize=False)
                text_embeds_bucket.append(text_embeds)
            text_embeds = torch.stack(text_embeds_bucket).mean(dim=0)
            text_embeds = F.normalize(text_embeds, dim=-1)
            self.text_cache.update(dict(zip(new_words, text_embeds)))

        cat_embeds = torch.stack([self.text_cache[word] for word in x])
        if w_bg:
            cat_embeds = torch.cat([cat_embeds, F.normalize(self.bg_embed, dim=-1)], dim=0)

        return cat_embeds
