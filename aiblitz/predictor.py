import os

import numpy as np
import pandas as pd
import tqdm

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from aiblitz.segment import segment_image, store_fen, parse_fen, idx_to_piece
from aiblitz.model import Net
from aiblitz.eval import evaluate


class BoardPredictionDataset(Dataset):

    def __init__(self, question, directory='train'):
        """
        :param question: string, Directory with all the images.
        """
        folder = "data/Q%d/%s" % (question, directory)
        self.images = sorted(os.listdir(folder), key=lambda x: int(x[:-4]))
        self.question = question
        self.directory = directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = "data/Q%d/%s/%s" % (self.question, self.directory, self.images[idx])
        segmented_image = segment_image(image_path)
        return segmented_image

    def __str__(self):
        return "Predicting Q%d-%s" % (self.question, self.directory)


def predict_one(model, image_frame):
    image = segment_image(image_frame)
    x = torch.stack([torch.from_numpy(image)])
    x = x.view(-1, 3, 32, 32).float()
    y = model(x)
    y = torch.argmax(y, -1)
    y = y.view(-1, 8, 8)
    return y


def predict(dataset):
    model = Net()
    model.load_state_dict(torch.load("weights/piece-recognizer.h5"))
    dataloader = DataLoader(dataset, batch_size=32)

    results = []
    with torch.no_grad():
        model.eval()
        iterator = tqdm.tqdm(dataloader)
        iterator.set_description(str(dataset))
        for x in iterator:
            x = x.view(-1, 3, 32, 32).float()
            y = model(x)
            y = torch.argmax(y, -1)
            y = y.view(-1, 8, 8)
            results.append(y.numpy())
    return np.concatenate(results)


def solve_1():
    dataset = BoardPredictionDataset(1, "test")
    result = predict(dataset)
    submission = []
    for idx, frame in enumerate(result):
        black_count = np.sum(frame < 6)
        white_count = np.sum(np.bitwise_and(6 <= frame, frame < 12))
        if white_count == black_count:
            print("Equality issue in Image", dataset.images[idx])
        submission.append("black" if white_count >= black_count else "white")
    with open("weights/result_1.csv", "w") as f:
        f.write("ImageID,label\n")
        for image_name, result in zip(dataset.images, submission):
            f.write(image_name[:-4] + "," + result + "\n")
        f.close()


def solve_2():
    dataset = BoardPredictionDataset(2, "test")
    result = predict(dataset)
    submission = []
    piece_to_val = {'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0,
                    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0, ' ': 0}

    for idx, frame in enumerate(result):
        total_score = 0
        for cell in np.reshape(frame, -1):
            total_score += piece_to_val[idx_to_piece[cell]]
        if total_score == 0:
            print("Equality issue in Image", dataset.images[idx])
        submission.append("white" if total_score >= 0 else "black")
    with open("weights/result_2.csv", "w") as f:
        f.write("ImageID,label\n")
        for image_name, result in zip(dataset.images, submission):
            f.write(image_name[:-4] + "," + result + "\n")
        f.close()


def solve_3():
    dataset = BoardPredictionDataset(3, "test")
    result = predict(dataset)
    submission = []
    for _idx, frame in enumerate(result):
        submission.append(store_fen(frame))
    with open("weights/result_3.csv", "w") as f:
        f.write("ImageID,label\n")
        for image_name, result in zip(dataset.images, submission):
            f.write(image_name[:-4] + "," + result + "\n")
        f.close()


def solve_5():
    for folder in ["test", "train", "val"]:
        dataset = BoardPredictionDataset(5, folder)
        result = predict(dataset)
        submission = []
        for _idx, frame in enumerate(result):
            submission.append(store_fen(frame))
        with open("weights/%s_5.csv" % folder, "w") as f:
            f.write("ImageID,label\n")
            for image_name, result in zip(dataset.images, submission):
                f.write(image_name[:-4] + "," + result + "\n")
            f.close()

    # dataset = pd.read_csv("weights/train_fen_5.csv")
    # moves = pd.read_csv("data/Q5/train.csv")["turn"].values
    # positions = dataset["label"].values
    # result = evaluate(zip(positions, moves))
    # print(result)


if __name__ == "__main__":
    # solve_1()
    # solve_2()
    # solve_3()
    solve_5()
