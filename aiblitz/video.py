import os
import cv2

from matplotlib import pyplot as plt
import torch

from aiblitz.model import Net
from aiblitz.predictor import predict_batch
from aiblitz.segment import store_fen

"""
Frames are read from mp4 file present at dir_in/filename.mp4
and the frames are extracted to dir_out/filename/frameN.jpg
The first frame is recorded at 0ms, and every next frame is
with a delay of DELAYms
"""


def extract_images(file_id, model, delay=100):
    # Get the predictions
    path = f"data/Q4/train/{file_id}.mp4"
    video_capture = cv2.VideoCapture(path)
    frames = []
    for index in range(100):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (index * delay))
        success, image_frame = video_capture.read()
        if not success:
            break
        frames.append(image_frame)
    results = predict_batch(model, frames)
    # Convert to FEN
    fen_list = []
    for result in results:
        fen = store_fen(result)
        if len(fen_list) < 1 or fen != fen_list[-1]:
            fen_list.append(fen)
    return fen_list


def predict_move(file_id, model):
    fen_list = extract_images(file_id, model)
    # TODO: Convert FEN-List to move-list
    print(fen_list)  # Dummy statement, write code here
    move_list = []
    return move_list


if __name__ == "__main__":
    network = Net()
    network.load_state_dict(torch.load("weights/piece-recognizer.h5"))
    print(predict_move(1, network))
