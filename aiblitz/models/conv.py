from ..agent import train, ChessPiecesDataset
import torch, torch.nn.functional, torch.utils.data


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 15, kernel_size=3)
        self.conv2_drop = torch.nn.Dropout2d()
        self.conv3 = torch.nn.Conv2d(15, 20, kernel_size=3)
        self.conv3_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(500, 50)
        self.fc2 = torch.nn.Linear(50, 13)

    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(self.conv2_drop(self.conv2(x)))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 500)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)


if __name__ == "__main__":
    network = Net()
    network.load_state_dict(torch.load("weights/piece-recognizer.h5"))
    train_dataset = torch.utils.data.dataloader.DataLoader(ChessPiecesDataset(3, 'train'), batch_size=32)
    val_dataset = torch.utils.data.dataloader.DataLoader(ChessPiecesDataset(3, 'val'), batch_size=32)
    train(network, None, train_dataset, epochs=1)
    train(network, None, val_dataset, epochs=1)
    # train(network, train_dataset, val_dataset)
