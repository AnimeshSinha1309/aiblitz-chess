import numpy as np
import pandas as pd
import tqdm

import torch, torch.nn.functional, torch.utils.data

from aiblitz.segment import segment_image, parse_fen, piece_to_idx


class ChessPiecesDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, question, directory='train', grayscale=False):
        """
        :param question: string, Directory with all the images.
        """
        df = pd.read_csv("data/Q%d/%s.csv" % (question, directory))
        self.images = df["ImageID"]
        self.labels = df["label"]
        self.question = question
        self.directory = directory
        self.grayscale = grayscale

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        parsed_fen = parse_fen(self.labels[idx])
        image_path = "data/Q%d/%s/%d.jpg" % (self.question, self.directory, self.images[idx])
        segmented_image = segment_image(image_path, grayscale=self.grayscale)
        return segmented_image, parsed_fen


def train(model, train_dataloader, val_dataloader, epochs=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.functional.nll_loss

    batch_losses, batch_accuracy = [], []
    val_batch_losses, val_batch_accuracy = [], []

    for epoch in range(epochs):
        if train_dataloader is not None:
            model.train()
            total_loss, total_samples, total_correct = 0, 0, 0
            train_iterator = tqdm.tqdm(train_dataloader)
            train_iterator.set_description("Training Epoch %d" % (epoch + 1))
            for x, y in train_iterator:
                optimizer.zero_grad()

                x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]).float()
                y = y.view(-1)
                out = model(x)

                loss = criterion(out, y)
                total_loss += loss.item()
                total_samples += y.shape[0]
                # noinspection PyUnresolvedReferences
                total_correct += (torch.max(out, 1)[1] == y).float().sum().item()
                loss.backward()
                optimizer.step()

                train_iterator.set_postfix(
                    loss=total_loss / total_samples,
                    accuracy=total_correct / total_samples)

            batch_losses.append(total_loss / total_samples)
            batch_accuracy.append(total_correct / total_samples)
            torch.save(model.state_dict(), "weights/piece-recognizer.h5")

        if val_dataloader is not None:
            total_loss, total_samples, total_correct = 0, 0, 0
            with torch.no_grad():
                model.eval()
                val_iterator = tqdm.tqdm(val_dataloader)
                val_iterator.set_description("Validation Epoch %d" % (epoch + 1))
                for x, y in val_iterator:
                    x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]).float()
                    y = y.view(-1)
                    out = model(x)
                    loss = criterion(out, y)
                    total_loss += loss.item()
                    total_samples += y.shape[0]
                    # noinspection PyUnresolvedReferences
                    total_correct += (torch.max(out, 1)[1] == y).float().sum().item()

                    val_iterator.set_postfix(
                        loss=total_loss / total_samples,
                        accuracy=total_correct / total_samples)

                val_batch_losses.append(total_loss / total_samples)
                val_batch_accuracy.append(total_correct / total_samples)


def get_piece(piece, inputs=tuple(range(100))):
    images = []
    for i in inputs:
        data = ChessPiecesDataset(3, 'train')
        image, label = data[i]
        image = image.reshape(-1, 3, 32, 32)
        label = label.reshape(-1)
        image = image[label == piece_to_idx[piece]]
        images.append(image)
    images = np.concatenate(images)
    return images.transpose((0, 2, 3, 1))
