# Instance Brownian Bridge as Texts for Open-vocabulary Video Instance Segmentation

## Getting Started

* [Dataset Prepare](datasets/README.md)
* [Installation](INSTALL.md)

## Training

1. Reproducing OpenVIS:
```bash
python train_net.py --config-file configs/openvoc_lvvis/openvis_R50_bs8_12000st.yaml --num-gpus 4
``` 

2. Reproducing OV2Seg:
```bash
python train_net.py --config-file configs/openvoc_lvvis/ov2seg_R50_bs8_12000st.yaml --num-gpus 4
```

3. Reproducing BriVIS:
```bash
python train_net.py --config-file configs/openvoc_lvvis/san_online_R50_bs8_12000st.yaml --num-gpus 4
python train_net.py --config-file configs/openvoc_lvvis/brivis_R50_bs8_12000st.yaml --num-gpus 4 MODEL.WEIGHTS work_dirs/openvoc_lvvis/san_online_R50_bs8_12000st/model_final.pth
```
