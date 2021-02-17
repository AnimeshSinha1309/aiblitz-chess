import os

import numpy as np
import pandas as pd
import tqdm

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from aiblitz.segment import segment_image, store_fen, idx_to_piece
from aiblitz.model import Net
from aiblitz.eval import evaluate, WHITE, BLACK
from aiblitz.video import predict_move

from sklearn.metrics import accuracy_score


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


def solve_4():
    videos = sorted(list(map(lambda x: int(x[:-4]), os.listdir("data/Q4/test"))))
    model = Net()
    model.load_state_dict(torch.load("weights/piece-recognizer.h5"))

    with open("weights/result_3.csv", "w") as f:
        f.write("VideoID,label\n")
        for video_name in tqdm.tqdm(videos):
            result = predict_move(video_name, model)
            f.write(str(video_name) + "," + " ".join(result) + "\n")
        f.close()


def pre_solve_5():
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


def make_labels_integer(seq):
    return [WHITE if x == "white" else BLACK for x in seq]


def make_labels_string(seq):
    return ["white" if x == WHITE else "black" for x in seq]


def solve_5_train():
    N = 1000

    train_csv = pd.read_csv("data/Q5/train.csv")
    moves = train_csv["turn"].values
    answers = train_csv["label"].values
    answers = make_labels_integer(answers)
    moves = make_labels_integer(moves)

    answers = answers[:N]

    dataset = pd.read_csv("weights/train_fen_5.csv")
    positions = dataset["label"].values

    positions, moves = positions[:N], moves[:N]
    result = evaluate(zip(positions, moves))
    # result = evaluate(zip(positions, moves))

    acc = accuracy_score(answers, result)

    print(f"Train accuracy: {acc}")

    # for i in range(N):
    #     if answers[i] != result[i]:
    #         print(f"Position: {i}")
    #         print(f"Answer: {answers[i]}")
    #         print(f"Board\n{positions[i]}")
    #         print(f"Move (official): {moves[i]}")


def solve_5_test():
    test_csv = pd.read_csv("data/Q5/test.csv", index_col=None)
    image_ids = test_csv["ImageID"].values
    moves = test_csv["turn"].values
    moves = make_labels_integer(moves)

    dataset = pd.read_csv("weights/test_fen_5.csv")
    positions = dataset["label"].values

    result = evaluate(zip(positions, moves))
    result = make_labels_string(result)

    cols = ["ImageID", "label"]
    df = pd.DataFrame({"ImageID": image_ids, "label": result}, columns=cols)
    df.to_csv("submit_5.csv", index=False)


if __name__ == "__main__":
    solve_4()
    # solve_5_train()
    # solve_5_test()
    # solve_1()
    # solve_2()
    # solve_3()
