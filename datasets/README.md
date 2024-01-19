
# Prepare Datasets

Expected dataset structure:

Dataset structure:
```
openvis
├── datasets
│   ├── ytvis_2019
│   │   ├── {train, valid, test}.json
│   │   ├── {train, valid, test}/
│   │   │   ├── JPEGImages/
│   │   │   ├── Annotations/
│   ├── ytvis_2021
│   │   ├── {train, valid, test}/
│   │   │   ├── JPEGImages/
│   │   │   ├── instances.json
│   ├── ovis
│   │   ├── annotations_{train, valid, test}.json/
│   │   ├── Images/
│   │   │   ├── {train, valid, test}/
│   ├── burst
│   │   ├── annotations/
│   │   │   ├── train/
│   │   │   │   ├── train.json
│   │   │   ├── {val, test}/
│   │   │   │   ├── all_classes.json
│   │   │   │   ├── common_classes.json
│   │   │   │   ├── uncommon_classes.json
│   │   ├── frames/
│   │   │   ├── {train, val, test}/
│   ├── lvvis
│   │   ├── {train, val}_instances.json
│   │   ├── {train, val}_ytvis_style.json
│   │   ├── {train, val}/
│   │   │   ├── JPEGImages/
│   ├── coco
│   │   ├── coco2ytvis2019_{train, val}.json
│   │   ├── coco2ytvis2021_{train, val}.json
│   │   ├── coco2ovis_{train, val}.json
│   │   ├── {train, val}2017/
│   │   ├── annotations/
│   │   │   ├── instances_{train, val}2017.json
```


## Youtube-VIS

```
openvis
├── datasets
│   ├── ytvis_2019
│   │   ├── {train, valid, test}.json
│   │   ├── {train, valid, test}/
│   │   │   ├── JPEGImages/
│   │   │   ├── Annotations/
│   ├── ytvis_2021
│   │   ├── {train, valid, test}/
│   │   │   ├── JPEGImages/
│   │   │   ├── instances.json
```

youtube-vis 2019:
1. register [2nd youtube vos challenge](https://competitions.codalab.org/competitions/20128)
2. download `{train,valid,test}.zip` from [youtube-vis 2019 frames google drive](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz)
3. download `{train,valid,test}.json` from [youtube-vis 2019 annotations google drive](https://drive.google.com/drive/folders/1VBeVXSf-HfrhyBurtu1xfGn6zQ-ALpjZ?usp=sharing)
4. unzip `{train,valid,test}.zip` to `datasets/ytvis_2019/{train,valid,test}/`
5. put `{train,valid,test}.json` to `datasets/ytvis_2019/{train,valid,test}.json`

youtube-vis 2021:
1. register [3rd youtube vos challenge](https://competitions.codalab.org/competitions/28988)
2. download `{train,valid,test}.zip` from [youtube-vis 2021 frames google drive](https://drive.google.com/drive/folders/12DxR2HWTVjULNwKVMdYAvhOmZ9gBxsX2)
3. unzip `{train,valid,test}.zip` to `datasets/ytvis_2021/{train,valid,test}/`


## OVIS

```
openvis
├── datasets
│   ├── ovis
│   │   ├── annotations_{train, valid, test}.json/
│   │   ├── Images/
│   │   │   ├── {train, valid, test}/
```

1. register [Occluded Video Instance Segmentation](https://codalab.lisn.upsaclay.fr/competitions/4763#participate)
2. download `{train,valid,test}.zip` from [OVIS frames google drive](https://drive.google.com/drive/folders/1eE4lLKCbv54E866XBVce_ebh3oXYq99b)
3. download `{train,valid,test}.json` from [OVIS annotations google drive](https://drive.google.com/drive/folders/1eE4lLKCbv54E866XBVce_ebh3oXYq99b)
4. unzip `{train,valid,test}.zip` to `datasets/ovis/Images/{train,valid,test}/`
5. put `annotations/{train,valid,test}.json` to `datasets/ovis/annotations_{train,valid,test}.json`


## BURST

```
openvis
├── datasets
│   ├── burst
│   │   ├── annotations/
│   │   │   ├── train/
│   │   │   │   ├── train.json
│   │   │   ├── {val, test}/
│   │   │   │   ├── all_classes.json
│   │   │   │   ├── common_classes.json
│   │   │   │   ├── uncommon_classes.json
│   │   ├── frames/
│   │   │   ├── {train, val, test}/
```

1. download videos (except AVA & HACS videos) from [TAO data](https://motchallenge.net/tao_download.php):
```
wget "https://motchallenge.net/data/1-TAO_TRAIN.zip" 
wget "https://motchallenge.net/data/2-TAO_VAL.zip" 
wget "https://motchallenge.net/data/3-TAO_TEST.zip"
unzip 1-TAO_TRAIN.zip
unzip 2-TAO_VAL.zip
unzip 3-TAO_TEST.zip
```
2. put uncompressed folders to `datasets/burst/frames/{train, val, test}`
2. sign in [MOTchallenge](https://motchallenge.net/login/) and download AVA & HACS videos
3. download annotations from [BURST-benchmark](https://github.com/Ali2500/BURST-benchmark): `wget https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip`
4. uncompress `annotations.zip` to `datasets/burst/annotations`


## LVVIS

```
openvis
├── datasets
│   ├── lvvis
│   │   ├── {train, val}_instances.json
│   │   ├── {train, val}_ytvis_style.json
│   │   ├── {train, val}/
│   │   │   ├── JPEGImages/
```

1. download videos from [LV-VIS github](https://github.com/haochenheheda/LVVIS);
    * download `train.zip` from [google drive](https://drive.google.com/file/d/1er2lBQLF75TI5O4wzGyur0YYoohMK6C3/view?usp=sharing);
    * download `val.zip` from [google drive](https://drive.google.com/file/d/1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn/view?usp=drive_link);
    * uncompressed `{train, val}.zip` to `datasets/lvvis/{train, val}/JPEGImages/`;
2. download annotations from [LV-VIS github](https://github.com/haochenheheda/LVVIS);
    * download `train_instances.json` from [google drive](https://drive.google.com/file/d/1k-o8gBMD7m1-fghw-a1iNDZCi2ZZgV9g/view?usp=sharing);
    * download `val_instances.json` from [google drive](https://drive.google.com/file/d/1stPD818M3gv7zUV3UIZG1Suru7Tk54jo/view?usp=sharing);
    * put `{train, val}_instances.json` to `datasets/lvvis/{train, val}_instances.json`;
3. convert annotations to youtube-vis style: 
```
python datasets/lvvis2ytvis.py
```


## COCO

```
$ROOT
├── datasets
│   ├── coco
│   │   ├── coco2ytvis2019_{train, val}.json
│   │   ├── coco2ytvis2021_{train, val}.json
│   │   ├── coco2ovis_{train, val}.json
│   │   ├── {train, val}2017/
│   │   ├── annotations/
│   │   │   ├── instances_{train, val}2017.json
```

1. download images from [COCO homepage](https://cocodataset.org/#home);
    * download `train2017.zip`, `val2017.zip`;
    * uncompress `{train, val}2017,zip` to `datasets/coco/{train, val}2017/`;
2. download annotations from [COCO homepage](https://cocodataset.org/#home);
    * download `annotations_trainval2017.zip`;
    * uncompress `annotations_trainval2017.zip` to `datasets/coco/annotations`;
3 convert annotations to youtube-vis style:
```
python datasets/coco2ytvis.py
```
