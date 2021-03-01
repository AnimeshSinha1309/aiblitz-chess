"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""

import numpy as np
import tqdm

import torch, torch.nn.functional


NUM_CLASSES = 13
NUM_ROUTING_ITERATIONS = 3
DEVICE = 'cpu'


def softmax(input_tensor, dim=1):
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    softmax_output = torch.nn.functional.softmax(
        transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmax_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(torch.nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = torch.nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = torch.nn.ModuleList(
                [torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
                 for _ in range(num_capsules)])

    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        outputs = None
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = torch.autograd.Variable(torch.zeros(*priors.size())).to(DEVICE)
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(torch.nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16 * NUM_CLASSES, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 784),
            torch.nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = torch.nn.functional.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = torch.nn.functional.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = torch.autograd.Variable(torch.eye(NUM_CLASSES)).to(DEVICE).index_select(
                dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(torch.nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = torch.nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = torch.nn.functional.relu(0.9 - classes, inplace=True) ** 2
        right = torch.nn.functional.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


def train(model, train_dataloader, val_dataloader, epochs=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = CapsuleLoss()

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
                x = augmentation(x.float() / 255.0, max_shift=2)
                y = y.view(-1)
                y = torch.eye(NUM_CLASSES).index_select(dim=0, index=y)

                x = torch.autograd.Variable(x).to(DEVICE)
                y = torch.autograd.Variable(y).to(DEVICE)
                print(x.shape, y.shape)
                out, reconstruction = model(x, y)
                loss = criterion(x, y, out, reconstruction)
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
            torch.save(model.state_dict(), "weights/capsule-network.h5")

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


if __name__ == "__main__":
    from aiblitz.agent import ChessPiecesDataset
    import torch.utils.data

    network = CapsuleNet()
    train_dataset = torch.utils.data.dataloader.DataLoader(
        ChessPiecesDataset(3, 'train', grayscale=True), batch_size=32)
    val_dataset = torch.utils.data.dataloader.DataLoader(
        ChessPiecesDataset(3, 'val', grayscale=True), batch_size=32)
    train(network, train_dataset, val_dataset)
