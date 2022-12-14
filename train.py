import torch
import torch.utils.data
from torchvision import datasets, transforms

from efficientff.labels import LabelsInjector
from efficientff.modules import FCNetFF


def train(
        n_layers: int,
        hidden_size: int,
        optimizer_name: str,
        lr: float,
        device: str,
        epochs: int,
        batch_size: int,
        theta: float,
        loss_fn_name: str
):
    """Train FCNetFF using MNISt dataset.
    """
    batch_size = batch_size // 2 # we will double the batch size to include negative examples
    input_len = 28 * 28 + len(datasets.MNIST.classes) # MNIST image size + number of classes
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
        batch_size=1, shuffle=True)

    # Define model
    model = FCNetFF([input_len] + [hidden_size] * n_layers, optimizer_name, {"lr": lr}, loss_fn_name)
    model.to(device)
    label_injector = LabelsInjector(datasets.MNIST.classes)

    # Train
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            input_data = label_injector.inject_train(data, target)
            input_data = input_data.to(device)
            signs = torch.cat([torch.ones(input_data.shape[0] // 2, device=device), - torch.ones(input_data.shape[0] // 2, device=device)])
            model.ff_train(input_data, signs, theta)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                input_data = label_injector.inject_eval(data)
                input_data = input_data.to(device)
                target = target.to(device)
                _, prob = model.positive_eval(input_data, theta)
                pred = prob.argmax(dim=0)
                correct += pred == target
        if isinstance(correct, torch.Tensor):
            correct = correct.item()
        print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # save model
    torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train FCNetFF')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=2000,
                        help='number of hidden units')
    parser.add_argument('--optimizer_name', type=str, default="Adam",
                        help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--device', type=str, default="cpu",
                        help='device to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--theta', type=float, default=2.,
                        help='theta parameter')
    parser.add_argument("--loss_fn_name", type=str, default="loss_fn")

    args = parser.parse_args()
    train(**vars(args))