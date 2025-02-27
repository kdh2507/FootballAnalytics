import glob
import cv2


root = "data/football_train"
video_paths = [path.replace(".mp4", "") for path in glob.iglob("")]