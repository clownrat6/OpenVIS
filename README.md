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
python train_net.py --config-file configs/openvoc_lvvis/openvis_R50_bs8_12000st.yaml --num-gpus 4
``` 

2. Reproducing [Towards Open-Vocabulary Video Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Towards_Open-Vocabulary_Video_Instance_Segmentation_ICCV_2023_paper.pdf):
```bash
python train_net.py --config-file configs/openvoc_lvvis/ov2seg_R50_bs8_12000st.yaml --num-gpus 4
```

3. Reproducing BriVIS:
```bash
python train_net.py --config-file configs/openvoc_lvvis/san_online_R50_bs8_12000st.yaml --num-gpus 4
python train_net.py --config-file configs/openvoc_lvvis/brivis_R50_bs8_12000st.yaml --num-gpus 4 MODEL.WEIGHTS work_dirs/openvoc_lvvis/san_online_R50_bs8_12000st/model_final.pth
```
