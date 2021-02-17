import pandas as pd
import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from aiblitz.segment import segment_image, parse_fen


class ChessPiecesDataset(Dataset):

    def __init__(self, question, directory='train'):
        """
        :param question: string, Directory with all the images.
        """
        df = pd.read_csv("data/Q%d/%s.csv" % (question, directory))
        self.images = df["ImageID"]
        self.labels = df["label"]
        self.question = question
        self.directory = directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        parsed_fen = parse_fen(self.labels[idx])
        image_path = "data/Q%d/%s/%d.jpg" % (self.question, self.directory, self.images[idx])
        segmented_image = segment_image(image_path)
        return segmented_image, parsed_fen


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 15, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(15, 20, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 13)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(model, train_dataloader, val_dataloader, epochs=10):
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = F.nll_loss

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

                x = x.view(-1, 3, 32, 32).float()
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
                    x = x.view(-1, 3, 32, 32).float()
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


if __name__ == "__main__":
    network = Net()
    network.load_state_dict(torch.load("weights/piece-recognizer.h5"))
    train_dataset = DataLoader(ChessPiecesDataset(3, 'train'), batch_size=32)
    val_dataset = DataLoader(ChessPiecesDataset(3, 'val'), batch_size=32)
    train(network, None, train_dataset, epochs=1)
    train(network, None, val_dataset, epochs=1)
    # train(network, train_dataset, val_dataset)
