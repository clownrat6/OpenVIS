import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.index import batch_index
from ..utils.distributed import concat_all_gather, get_rank


def brownian_bridge_distance(embed, bridge):
    # embed: (b, t, c)
    # bridge: (b, 3)
    embed = batch_index(embed, bridge)
    bh, bp, bt = bridge[..., 0:1], bridge[..., 1:2], bridge[..., 2:3]
    alpha = torch.true_divide(bp-bh, bt-bh) 
    sigma = alpha * (bt-bp)

    x = embed[..., 1, :] - (1 - alpha) * embed[..., 0, :] - alpha * embed[..., 2, :]

    dist = -torch.norm(x, p=2, dim=-1, keepdim=True)**2 / (2 * sigma**2)
    dist = dist.squeeze(dim=-1)

    return torch.exp(dist)


def neg_brownian_bridge_distance(embed, neg_embed, bridge, topk=5):
    # embed: (b, t, c)
    # neg_embed: (q, t, c)
    # bridge：(b, 3)
    pos_len, _, _ = embed.shape
    neg_len, _, _ = neg_embed.shape

    neg_embed = torch.cat([embed, neg_embed])
    neg_len = neg_len + pos_len

    embed = batch_index(embed, bridge)
    embed = embed[:, None].repeat(1, neg_len, 1, 1)

    bridge = bridge[:, None].repeat(1, neg_len, 1)

    neg_embed = neg_embed[None, :] .repeat(pos_len, 1, 1, 1)
    neg_embed = batch_index(neg_embed.flatten(0, 1), bridge.flatten(0, 1)).reshape(pos_len, neg_len, 3, -1)

    bh, bp, bt = bridge[..., 0:1], bridge[..., 1:2], bridge[..., 2:3]
    alpha = torch.true_divide(bp - bh, bt - bh) 
    sigma = alpha * (bt - bp)

    x = neg_embed[..., 1, :] - (1 - alpha) * embed[..., 0, :] - alpha * embed[..., 2, :]

    dist = -torch.norm(x, p=2, dim=-1, keepdim=True)**2 / (2 * sigma**2)
    dist = dist.squeeze(dim=-1)

    if topk != None:
        self_dist = dist[torch.arange(pos_len, device=embed.device), torch.arange(pos_len, device=embed.device)]
        dist[torch.arange(pos_len, device=embed.device), torch.arange(pos_len, device=embed.device)] = -10000
        dist, ids = torch.topk(dist, k=5, dim=-1)
        dist = torch.cat([self_dist[:, None], dist], dim=-1)

    return torch.exp(dist)


class BrownianBridgeCriterion(nn.Module):

    def __init__(self, hidden_dim=256, proj_dim=256):
        super().__init__()
        self.brownian_proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, frame_embeds, delta=0.3):
        bs, t, q, c = frame_embeds.shape

        # shape: (b, t, q, c)
        frame_embeds = self.brownian_proj(frame_embeds)
        frame_embeds_all = concat_all_gather(frame_embeds)
        rank = get_rank()
        # shape: (b, t, q, c)
        cur_embeds = frame_embeds
        # shape: (b, t, q, c) cat (b * (n - 1), t, q, c)
        # move embeds of each rank to first order
        other_embeds = torch.cat([frame_embeds_all[:rank * bs], frame_embeds_all[(rank + 1) * bs:]])

        # shape: (b, t, q, c) -> (b, q, t, c) -> (bq, t, c)
        cur_embeds = cur_embeds.permute(0, 2, 1, 3).flatten(0, 1)
        other_embeds = other_embeds.permute(0, 2, 1, 3).flatten(0, 1)

        cur_embeds = F.normalize(cur_embeds, dim=-1)
        other_embeds = F.normalize(other_embeds, dim=-1)
        n, t = cur_embeds.shape[:2]

        # random select bridge contrast index
        bridge = torch.randint(1, t - 1, (n, 3), device=frame_embeds.device)
        bridge[:, 0] = 0
        bridge[:, 2] = t - 1
        numer = brownian_bridge_distance(cur_embeds, bridge)
        deno = neg_brownian_bridge_distance(cur_embeds, other_embeds, bridge, topk=5)

        brownian_loss = numer / deno.sum(dim=-1)

        # head-tail matching loss
        score = torch.einsum('nc,nc->n', cur_embeds[:, 0], cur_embeds[:, -1])
        head_tail_match = nn.Softplus()(delta - score).mean()

        brownian_loss = brownian_loss.mean()

        return brownian_loss, head_tail_match