import torch
import torch.utils.data
from torchvision import datasets, transforms

from efficientff.modules import RecurrentFCNetFF


def get_dataloader(batch_size: int):
    batch_size = batch_size // 2  # we will double the batch size to include negative examples
    # Load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=False)
    return train_loader, test_loader


def train(n_layers: int, hidden_size: int, epochs: int, batch_size: int, loss_fn: str, lr: float, theta: float, device: str):
    optimizer_name = "Adam"
    optimizer_args = {"lr": lr}
    input_len = 28 * 28  # MNIST image size + number of classes
    train_loader, test_loader = get_dataloader(batch_size)
    # Define model
    layer_sizes = [input_len] + [hidden_size] * n_layers + [len(datasets.MNIST.classes)]
    model = RecurrentFCNetFF(layer_sizes, optimizer_name, optimizer_args, loss_fn).to(device)
    for epoch in range(epochs):
        accumulated_goodness = None
        model.train()
        for j, (data, target) in enumerate(train_loader):
            data = data.to(device).reshape(-1, 28 * 28)
            target = torch.functional.F.one_hot(target.to(device), num_classes=len(datasets.MNIST.classes))
            _, goodness = model.ff_train(data, target, theta)
            if accumulated_goodness is None:
                accumulated_goodness = goodness
            else:
                accumulated_goodness[0] += goodness[0]
                accumulated_goodness[1] += goodness[1]
        print(f"Epoch {epoch + 1}")
        print(f"Accumulated goodness: {accumulated_goodness}")
        print(f"Goodness ratio: {(accumulated_goodness[0] - accumulated_goodness[1]) / abs(max(accumulated_goodness))}")
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device).reshape(-1, 28 * 28)
                target = target.to(device)
                pred, _ = model.positive_eval(data, theta)
                correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Test accuracy: {correct} / 10000 ({correct / 10000 * 100}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--loss_fn", type=str, default="alternative_loss_fn")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--theta", type=float, default=2.)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(**vars(args))