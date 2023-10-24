import os

import cv2
import mmcv

def split_single_video(video_path, frames_dir=""):
	cap = cv2.VideoCapture(video_path)
	cnt = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			success, buffer = cv2.imencode(".png", frame)
			if success:
				with open(f"{frames_dir}{cnt}.png", "wb") as f:
					f.write(buffer.tobytes())
					f.flush()
				cnt += 1
		else:
			break


def process(paths):
    v_path, v_fram_dir = paths
    if not os.path.exists(v_fram_dir):
    	os.makedirs(v_fram_dir, 0o775)
    split_single_video(v_path, v_fram_dir)


# rename with the directory where you stored videos
VIDEOS_DIR = "./datasets/uvo/uvo_videos_dense/"
# rename with the directory where you would like to store frames
FRAMES_DIR = "./datasets/uvo/uvo_videos_dense_frames/"

if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR, 0o775)

all_v_paths = os.listdir(VIDEOS_DIR)
records = [(f"{VIDEOS_DIR}{v_path}", f"{FRAMES_DIR}{v_path[:-4]}/") for v_path in all_v_paths]
mmcv.track_parallel_progress(process, records, 16)
