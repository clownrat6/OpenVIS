# Installation

- python >= 3.6
- pytorch 1.10 & torchvision: Refer to [pytorch.org](https://pytorch.org)
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```
- cuda 10.2
- detectron2: Refer to [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html)
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
```
- other packages
```
pip install -r requirements.txt
```
- clip
```
pip install git+https://github.com/openai/CLIP.git
```
- mask adapted clip
```
pip install -e third_parties/mask_adapted_clip 
```
- CUDA kernel for `MSDeformAttn`
```
cd openvis/modeling/pixel_decoder/ops
bash make.sh
```
- trackeval: Refer to [TrackEval](https://github.com/JonathonLuiten/TrackEval)
```
pip install git+https://github.com/sennnnn/TrackEval.git
```
