import resource

import torch
import torch.utils.data
from forward_forward import train_with_forward_forward_algorithm


def train(
        n_layers: int,
        hidden_size: int,
        lr: float,
        device: str,
        epochs: int,
        batch_size: int,
        theta: float,
        save_memory_profile: str = None,
):
    """Train FCNetFF using MNISt dataset.
    """
    batch_size = batch_size // 2  # we will double the batch size to include negative examples
    train_with_forward_forward_algorithm(
        model_type="progressive",
        n_layers=n_layers,
        hidden_size=hidden_size,
        lr=lr,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        theta=theta,
    )
    if save_memory_profile is not None:
        if torch.cuda.is_available() and "cuda" in device:
            memory_allocated = torch.cuda.max_memory_allocated(device=device)
        else:
            memory_allocated = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(save_memory_profile, "w") as f:
            f.write(f"{memory_allocated}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train FCNetFF')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--device', type=str, default="cpu",
                        help='device to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--theta', type=float, default=2.,
                        help='theta parameter')
    parser.add_argument("--save_memory_profile", type=str, default=None)
    args = parser.parse_args()
    train(**vars(args))
