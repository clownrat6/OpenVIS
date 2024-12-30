import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap

fontsize = 25
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = fontsize

def hextorgb(num):
    num = num[1:]
    return (int(num[:2], 16), int(num[2:4], 16), int(num[4:6], 16))

def gradual(src, dst, steps=10):
    src = np.array(src).astype(np.float32) / 256
    dst = np.array(dst).astype(np.float32) / 256
    delta = (dst - src) / steps
    res = [src]
    for i in range(steps):
        res.append(res[-1] + delta)
    return np.array(res)

def make_colormap(src_color, dst_color):
    steps = 20
    src_color_rgb = hextorgb(src_color)
    dst_color_rgb = hextorgb(dst_color)
    gradual_res = np.array(gradual(src_color_rgb, dst_color_rgb, steps=steps))

    color_bar = tuple([(1 / steps * i, tuple(gradual_res[i])) for i in range(steps + 1)])

    ccolormap = LinearSegmentedColormap.from_list('ccolormap', color_bar)

    return ccolormap

def draw_rela(dataset, save_name, cmap='Purple', color='#7669AF'):
    # fig = plt.figure(figsize=(4, 4.5), dpi=300)

    # g = sns.jointplot(
    #     x='dIoU',
    #     y='IoU',
    #     data=dataset,
    #     kind='kde',
    #     space=0,
    #     fill=True,
    #     # cbar=True,
    #     cmap="Blues",  # make_colormap(src_color, dst_color)
    #     n_levels=10,
    #     color="#1460A8",
    # )

    # g = sns.kdeplot(dataset, x='IoU', y='Length', fill=True, cmap=cmap)

    g = sns.jointplot(
        dataset,
        x='Length',
        y='IoU',
        kind='kde',
        space=0,
        fill=True,
        # cbar=True,
        n_levels=10,
        cmap=cmap,
        ylim=[0, 1.0],
        xlim=[25, 35],
        color=color,
    )

    g.ax_joint.set_xlabel('Length', fontsize=fontsize, fontweight='bold')
    g.ax_joint.set_ylabel('IoU', fontsize=fontsize, fontweight='bold')

    # g.ax_joint.set_xlim([23, 48])
    # g.ax_joint.set_xticks([25, 30, 35, 40 ,45])
    # g.ax_joint.set_xticklabels([25, 30, 35, 40 ,45], fontsize=fontsize - 2)

    g.ax_joint.set_xlim([24, 36])
    g.ax_joint.set_xticks([25, 27, 29, 31, 33, 35])
    g.ax_joint.set_xticklabels([25, 27, 29, 31, 33, 35], fontsize=fontsize - 2)

    g.ax_joint.set_ylim([-0.15, 1.15])
    g.ax_joint.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    g.ax_joint.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=fontsize - 2)

    g.ax_joint.tick_params(bottom=False, top=False, left=False, right=False)
    g.ax_marg_x.tick_params(bottom=False, top=False, left=False, right=False)
    g.ax_marg_y.tick_params(bottom=False, top=False, left=False, right=False)

    # sns.set_style('darkgrid')

    g.ax_joint.set(xlabel=None, ylabel=None)

    g.fig.set_figwidth(4.5)
    g.fig.set_figheight(4.8)
    sns.set_style('whitegrid')
    plt.tight_layout(pad=0)
    plt.savefig(f'{save_name}.png')
    plt.savefig(f'{save_name}.pdf')

# lens = np.load('cvpr2024/density/mean_brow_len_collect.npy')
# ious = np.load('cvpr2024/density/mean_brow_iou_collect.npy')

lens = np.load('cvpr2024/density/ytvis_2019_mean_dire_lens.npy')
ious = np.load('cvpr2024/density/ytvis_2019_mean_dire_ious.npy')

cond = (lens > 25) * (ious > 0.05)
lens = lens[cond]
ious = ious[cond]

dataset = {
    'IoU': ious,
    'Length': lens,
}

dataset = pd.DataFrame(dataset)

draw_rela(dataset, 'cvpr2024/density/iou_len_brow', cmap='Greens', color='#39A257')
