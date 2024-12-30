import json
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

linewidth = 3
colors = ['#78D151', '#48AD8A', '#26788E', '#3C3F84']
# colors = ['#211951', '#836FFF', '#15F5BA', '#F0F3FF']
# colors = ['#00B050', '#7030A0']
font_size = 20
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.family'] = 'Times New Roman'


def kde_draw(ious, colors, label_names):
    fig = plt.figure(figsize=(9, 4.5), dpi=300)

    # ious[0] = [x - random.random() for x in ious[0]]
    # ious[1] = [x + random.random() * 0.1 for x in ious[1]]
    # ious[2] = [x + random.random() * 0.1 for x in ious[2]]
    ious[3] = [x + random.random() for x in ious[3]]

    ax = fig.add_subplot(111)
    gs = [sns.kdeplot(ious[i], color=colors[i], linewidth=0, label=label_names[i], clip=(0.0, 5.0)) for i in range(len(ious))]
    # g1 = sns.kdeplot(ious[0], color=colors[0], linewidth=0, label=label_names[0], clip=(0.0, 5.0))
    # g2 = sns.kdeplot(ious[1], color=colors[1], linewidth=0, label=label_names[1], clip=(0.0, 5.0))
    # g3 = sns.kdeplot(ious[2], color=colors[2], linewidth=0, label=label_names[2], clip=(0.0, 5.0))

    # g.set(ylabel='Density', xlabel='IoU')
    # ax.set_xlabel('Entropy', fontsize=font_size, fontweight='bold')
    # ax.set_ylabel('Density', fontsize=font_size, fontweight='bold')
    ax.set_ylim([0, 0.32])
    ax.set_xlim([-0.2, 5.2])
    # #adjust y-axis label position
    # ax.yaxis.set_label_coords(-.1, .5)

    # #adjust x-axis label position 
    # ax.xaxis.set_label_coords(.5, -.1)

    ax.set(xlabel=None, ylabel=None)

    ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30'], fontsize=font_size - 2)

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=font_size - 2)

    def draw_dist(iou, color, g):
        kde = gaussian_kde(iou)
        # get the min and max of the x-axis
        xmin, xmax = g.get_xlim()
        # create points between the min and max
        x = np.linspace(xmin, xmax, 1000)
        # calculate the y values from the model
        kde_y = kde(x)
        x0 = x[(x >= 0) * (x <= 5)]
        y0 = kde_y[(x >= 0) * (x <= 5)]
        # fill the areas
        g.fill_between(x=x0, y1=y0, color=color, alpha=.4)
        p = ax.plot([x0[0]] + x0.tolist() + [x0[-1]], [0] + y0.tolist() + [0], color=color, linewidth=linewidth)[0]

    draw_dist(ious[0], colors[0], gs[0])
    draw_dist(ious[1], colors[1], gs[1])
    draw_dist(ious[2], colors[2], gs[2])
    draw_dist(ious[3], colors[3], gs[3])

    plt.scatter(4.6, 0.14, s=30, marker='o', color=colors[-1])
    plt.plot([4.6, 4.6], [0.14, 0.20], color=colors[-1], linewidth=3)
    plt.text(4.2, 0.21, 'Vanilla', color=colors[-1], fontsize=font_size, fontweight='bold')

    plt.scatter(3.2, 0.15, s=30, marker='o', color=colors[-2])
    plt.plot([3.2, 3.2], [0.15, 0.24], color=colors[-2], linewidth=3)
    plt.text(2.9, 0.25, 'IDOL', color=colors[-2], fontsize=font_size, fontweight='bold')

    plt.scatter(2, 0.23, s=30, marker='o', color=colors[-3])
    plt.plot([2, 2], [0.23, 0.26], color=colors[-3], linewidth=3)
    plt.text(1.64, 0.27, 'CTVIS', color=colors[-3], fontsize=font_size, fontweight='bold')

    plt.scatter(0.5, 0.15, s=30, marker='o', color=colors[-4])
    plt.plot([0.5, 0.5], [0.15, 0.21], color=colors[-4], linewidth=3)
    plt.text(0.29, 0.22, 'BTA', color=colors[-4], fontsize=font_size, fontweight='bold')

    # plt.scatter(1.2, 0.45, s=30, marker='o', color=colors[2])
    # plt.plot([1.2, 1.8], [0.45, 0.45], color=colors[2], linewidth=3)
    # plt.text(1.9, 0.436, 'Bridge-Text', color=colors[2], fontsize=font_size, fontweight='bold')

    # kde1 = gaussian_kde(ious[0])
    # # get the min and max of the x-axis
    # xmin, xmax = g1.get_xlim()
    # # create points between the min and max
    # x = np.linspace(xmin, xmax, 1000)
    # # calculate the y values from the model
    # kde_y = kde1(x)
    # x0 = x[(x >= 0) * (x <= 5)]
    # y0 = kde_y[(x >= 0) * (x <= 5)]
    # # fill the areas
    # g1.fill_between(x=x0, y1=y0, color=colors[0], alpha=.4)
    # p1 = ax.plot([x0[0]] + x0.tolist() + [x0[-1]], [0] + y0.tolist() + [0], color=colors[0], linewidth=linewidth)[0]

    # plt.scatter(4.1, 0.10, s=30, marker='o', color=colors[0])
    # plt.plot([4.1, 4.1], [0.10, 0.20], color=colors[0], linewidth=3)
    # plt.text(3.5, 0.22, 'Frame-Text', color=colors[0], fontsize=font_size, fontweight='bold')

    # kde2 = gaussian_kde(ious[1])
    # # get the min and max of the x-axis
    # xmin, xmax = g2.get_xlim()
    # # create points between the min and max
    # x = np.linspace(xmin, xmax, 1000)
    # # calculate the y values from the model
    # kde_y = kde2(x)
    # x0 = x[(x >= 0) * (x <= 5)]
    # y0 = kde_y[(x >= 0) * (x <= 5)]
    # # fill the areas
    # g2.fill_between(x=x0, y1=y0, color=colors[1], alpha=.4)
    # p2 = ax.plot([x0[0]] + x0.tolist() + [x0[-1]], [0] + y0.tolist() + [0], color=colors[1], linewidth=linewidth)[0]
    # # ax.plot(x0, y0, color=colors[1], linewidth=linewidth)

    # plt.scatter(1.8, 0.35, s=30, marker='o', color=colors[1])
    # plt.plot([1.8, 2.4], [0.35, 0.35], color=colors[1], linewidth=3)
    # plt.text(2.5, 0.336, 'Frame-Text w/ TIR', color=colors[1], fontsize=font_size, fontweight='bold')

    # kde3 = gaussian_kde(ious[2])
    # # get the min and max of the x-axis
    # xmin, xmax = g3.get_xlim()
    # # create points between the min and max
    # x = np.linspace(xmin, xmax, 1000)
    # # calculate the y values from the model
    # kde_y = kde3(x)
    # x0 = x[(x >= 0) * (x <= 5)]
    # y0 = kde_y[(x >= 0) * (x <= 5)]
    # # fill the areas
    # g3.fill_between(x=x0, y1=y0, color=colors[2], alpha=.4)
    # p3 = ax.plot([x0[0]] + x0.tolist() + [x0[-1]], [0] + y0.tolist() + [0], color=colors[2], linewidth=linewidth)[0]

    # plt.scatter(1.2, 0.45, s=30, marker='o', color=colors[2])
    # plt.plot([1.2, 1.8], [0.45, 0.45], color=colors[2], linewidth=3)
    # plt.text(1.9, 0.436, 'Bridge-Text', color=colors[2], fontsize=font_size, fontweight='bold')

    # plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
    # plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # plt.plot([0.5, 0.5], [0, 4], linewidth=1.5, linestyle='--', c='red')
    # plt.text(
    #     0.05,
    #     0.175,
    #     f'{label_name}',
    #     fontsize=font_size + 5,
    #     fontweight='bold',
    # )
    # plt.text(
    #     -0.1,
    #     3.2,
    #     f'IoU > 0.5: {iou_rate * 100.0:.1f}%',
    #     fontsize=font_size + 2,
    #     fontweight='bold',
    #     # color='red'
    # )
    # plt.text(0.46, -0.04, 'IoU', fontsize=font_size + 2, fontweight='bold')
    # plt.text(-.05, 0.26, 'Rate', fontsize=font_size + 2, fontweight='bold')
    plt.tight_layout(pad=0)

    # leg = plt.legend(
    #     handles=[p1, p2, p3],
    #     labels=['Frame-wise', 'TFrame-wise', 'Bridge-wise'],
    #     labelspacing=0.8,
    #     # columnspacing=0.3,
    #     handletextpad=0.3,
    #     prop={
    #         'weight': 'bold',
    #         'size': font_size - 2,
    #     },
    #     loc='upper right',
    # )

    return fig

# bta: swin/san_online_SwinB_bs16_6000st_ViT-L-336/model_0001999.pth
results = json.load(open('./cvpr2024/entropy/swin_san_online_best_bta.json', 'r'))
entropys1 = []
for res in results:
    entropys1.append(res['entropy'])

# idol: brivis_v2_R50_bs16_6000st/model_0004999.pth
results = json.load(open('./cvpr2024/entropy/brivis_v2_idol.json', 'r'))

entropys2 = []
for res in results:
    entropys2.append(res['entropy'])

# ctvis: san_online_coco1.0/model_0002999.pth
results = json.load(open('./cvpr2024/entropy/san_online_coco1.0_ctvis.json', 'r'))

entropys3 = []
for res in results:
    entropys3.append(res['entropy'])

# baseline: san_online_R50_bs16_6000st/model_0000999.pth
results = json.load(open('./cvpr2024/entropy/san_online_worse_baseline.json', 'r'))
entropys4 = []
for res in results:
    entropys4.append(res['entropy'])

# entropys2 = []
# for seq in results['sequences']:
#     segs = seq['segmentations']
#     track_ids = []
#     for seg in segs:
#         seg_ids = seg.keys()
#         for seg_id in seg_ids:
#             if seg_id not in track_ids:
#                 entropys2.append(seg[seg_id]['entropy'])
#                 track_ids.append(seg_id)

results = json.load(open('work_dirs/openvoc_ytvis_2019/san_online_R50_bs16_6000st_coco1.0/inference/results.json', 'r'))

entropys3 = []
for res in results:
    entropys3.append(res['entropy'])

fig = kde_draw([entropys1, entropys2, entropys3, entropys4], colors, ['Baseline', 'TBaseline', 'OpenBridgeVIS', 'OpenBridgeVIs'])
fig.savefig('cvpr2024/entropy/entropy_dist.png')
fig.savefig('cvpr2024/entropy/entropy_dist.pdf')
