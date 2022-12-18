import torch
from pytorch_memlab import LineProfiler, MemReporter
from torchvision import datasets

from efficientff.modules import FCNetFFProgressive
from train_base import train as train_ff_mnist
from train_base import get_dataloader as get_dataloader_mnist
from train_recurrent import train as train_recurrent_mnist

def run_backprop_mnist(
        n_layers: int,
        hidden_size: int,
        lr: float,
        device: str,
        epochs: int,
        batch_size: int,
    ):
    optimizer_kwargs = {"lr": lr}
    loss_fn = torch.nn.CrossEntropyLoss()
    layer_sizes = [28 * 28] + [hidden_size] * n_layers
    model = FCNetFFProgressive(layer_sizes=layer_sizes, optimizer_name="Adam", optimizer_kwargs=optimizer_kwargs, epochs=epochs, loss_fn_name="loss_fn")
    to_class_layer = torch.nn.Linear(layer_sizes[-1], 10)
    backprop_model = torch.nn.Sequential(model, to_class_layer).to(device)
    train_loader, test_loader = get_dataloader_mnist(batch_size=batch_size)
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


def main():
    n_layers = 3
    hidden_size = 2000
    lr = 0.03
    device = "cuda:0"
    epochs = 1
    batch_size = 5000
    theta = 2.0
    reporter_backprop = MemReporter(run_backprop_mnist)
    run_backprop_mnist(
        n_layers=n_layers, 
        hidden_size=hidden_size, 
        lr=lr, 
        device=device, 
        epochs=epochs, 
        batch_size=batch_size,
    )
    reporter_ff = MemReporter(train_ff_mnist)
    train_ff_mnist(
        n_layers=n_layers, 
        hidden_size=hidden_size, 
        lr=lr, 
        optimizer_name="Adam",
        loss_fn_name="alternative_loss_fn",
        device=device, 
        epochs=epochs, 
        batch_size=batch_size, 
        theta=theta
    )
    reporter_recurrent = MemReporter(train_recurrent_mnist)
    train_recurrent_mnist(
        n_layers=n_layers,
        hidden_size=hidden_size,
        lr=lr,
        loss_fn="alternative_loss_fn",
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        theta=theta
    )
    reporter_backprop.report()
    reporter_ff.report()
    reporter_recurrent.report()

if __name__ == "__main__":
    main()
