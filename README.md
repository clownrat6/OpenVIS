# Instance Brownian Bridge as Texts for Open-vocabulary Video Instance Segmentation

## Getting Started

* Download [Mask2former R50 weights trained on COCO Instance Segmentation](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) and put it in `pretrained/model_final_3c8ec9.pkl`
* [Dataset Prepare](datasets/README.md)
* [Installation](INSTALL.md)

## Updates

* [**2024.1.21**] Model ZOO is in preparetion. If your have any problems about this codebase, please contact me `cyanlaser@stu.pku.edu.cn`

## Training

1. Reproducing [OpenVIS: Open-vocabulary Video Instance Segmentation](https://arxiv.org/pdf/2305.16835.pdf):
```bash
python train_net.py --config-file configs/openvoc_ytvis/openvis_R50_bs16_6000st.yaml --num-gpus 8
``` 

2. Reproducing BriVIS:
```bash
python train_net.py --config-file configs/openvoc_ytvis_coco/san_online_R50_bs16_6000st.yaml --num-gpus 8
python train_net.py --config-file configs/openvoc_ytvis_coco/brivis_R50_bs16_6000st.yaml --num-gpus 8 MODEL.WEIGHTS work_dirs/openvoc_ytvis_coco/san_online_R50_bs16_6000st/model_final.pth
```
