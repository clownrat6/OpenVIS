
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
│   ├── ovis
│   │   ├── annotations_{train, valid, test}.json/
│   │   ├── Images/
│   │   │   ├── {train, valid, test}/
│   ├── uvo
│   │   ├── uvo_videos_dense/
│   │   ├── uvo_videos_dense_frames/
│   │   ├── VideoDenseSet/
│   │   │   ├── UVO_video_train_dense_with_label.json
│   │   │   ├── UVO_video_val_dense_with_label.json
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


## UVO

```
openvis
├── datasets
│   ├── uvo
│   │   ├── uvo_videos_dense/
│   │   ├── uvo_videos_dense_frames/
│   │   ├── VideoDenseSet/
│   │   │   ├── UVO_video_train_dense_with_label.json
│   │   │   ├── UVO_video_val_dense_with_label.json
```

1. Download pre-processed videos (`uvo_videos_dense.zip`) from [Tarun Kalluri's google drive](https://drive.google.com/drive/folders/1fOhEdHqrp_6D_tBsrR9hazDLYV2Sw1XC)
2. Unzip `uvo_videos_dense.zip` to `datasets/uvo/uvo_videos_dense`
3. Convert videos to frames by: `python datasets/uvo_video2frames.py` and the frames are storaged in `datasets/uvo/uvo_videos_dense_frames`;
4. Download annotations (`VideoDenseSet`) from [UVO v1.0 google drive](https://drive.google.com/drive/folders/1HqEX_bJ9k0qNf9jw2D6RG1X8jHh4_7GH)
5. Put `VideoDenseSet` to `datasets/uvo/VideoDenseSet`


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
    * download `train_instances.json` from [google drive](https://drive.google.com/file/d/1er2lBQLF75TI5O4wzGyur0YYoohMK6C3/view?usp=sharing);
    * download `val_instances.json` from [google drive](https://drive.google.com/file/d/1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn/view?usp=drive_link);
    * put `{train, val}_instances.json` to `datasets/lvvis/{train, val}_instances.json`;
3. convert annotations to youtube-vis style: 
```
python datasets/lvvis2ytvis.py
```
