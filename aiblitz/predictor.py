import os

import numpy as np
import tqdm

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from aiblitz.segment import segment_image, store_fen, idx_to_piece
from aiblitz.model import Net


class BoardPredictionDataset(Dataset):

    def __init__(self, question, directory='train'):
        """
        :param question: string, Directory with all the images.
        """
        folder = "data/Q%d/%s" % (question, directory)
        self.images = sorted(os.listdir(folder))
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
            x = x.view(-1, 1, 32, 32).float()
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


if __name__ == "__main__":
    solve_3()