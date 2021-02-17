import os
import cv2

from matplotlib import pyplot as plt
import torch

from aiblitz.model import Net

"""
Frames are read from mp4 file present at dir_in/filename.mp4
and the frames are extracted to dir_out/filename/frameN.jpg
The first frame is recorded at 0ms, and every next frame is
with a delay of DELAYms
"""
DELAY = 100  # in milliseconds
model = Net()
model.load_state_dict(torch.load("weights/piece-recognizer.h5"))
model.eval()


def extract_images(file_id):
    path = f"data/Q4/train/{file_id}.mp4"
    video_capture = cv2.VideoCapture(path)
    for index in range(100):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (index * DELAY))
        success, image_frame = video_capture.read()
        if not success:
            break
        plt.imshow(image_frame)
        plt.show()


if __name__ == "__main__":
    extract_images(1)
