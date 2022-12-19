import argparse
import resource

import torch
import torch.utils.data
from forward_forward.utils.modules import FCNetFFProgressive
from torchvision import datasets, transforms


def get_dataloader(batch_size: int):
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
        batch_size=1000, shuffle=True)
    return train_loader, test_loader


def run_backprop_mnist(
        n_layers: int,
        hidden_size: int,
        lr: float,
        device: str,
        epochs: int,
        batch_size: int,
        save_memory_profile: str = None,
    ):
    optimizer_kwargs = {"lr": lr}
    loss_fn = torch.nn.CrossEntropyLoss()
    layer_sizes = [28 * 28] + [hidden_size] * n_layers
    model = FCNetFFProgressive(layer_sizes=layer_sizes, optimizer_name="Adam", optimizer_kwargs=optimizer_kwargs, epochs=epochs, loss_fn_name="loss_fn")
    to_class_layer = torch.nn.Linear(layer_sizes[-1], 10)
    backprop_model = torch.nn.Sequential(model, to_class_layer).to(device)
    train_loader, test_loader = get_dataloader(batch_size=batch_size)
    optimizer = torch.optim.Adam(backprop_model.parameters(), **optimizer_kwargs)
    for _ in range(epochs):
        backprop_model.train()
        for j, (data, target) in enumerate(train_loader):
            data = data.reshape(-1, 28 * 28).to(device)
            target = target.to(device)
            pred = backprop_model(data)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        backprop_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.reshape(-1, 28 * 28).to(device)
                target = target.to(device)
                pred = backprop_model(data)
                correct += pred.argmax(dim=1).eq(target).sum().item()
        print(f"Accuracy: {correct / len(test_loader.dataset)}")
    if save_memory_profile is not None:
        if torch.cuda.is_available() and "cuda" in device:
            memory_allocated = torch.cuda.max_memory_allocated(device=device)
        else:
            memory_allocated = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(save_memory_profile, "w") as f:
            f.write(f"{memory_allocated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_memory_profile", type=str, default=None)
    args = parser.parse_args()
    run_backprop_mnist(**vars(args))
