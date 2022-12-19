import resource

import torch
import torch.utils.data
from forward_forward import train_with_forward_forward_algorithm


def train(
        n_layers: int,
        hidden_size: int,
        epochs: int,
        batch_size: int,
        lr: float,
        theta: float,
        device: str,
        save_memory_profile: str = None
):
    batch_size = batch_size // 2  # we will double the batch size to include negative examples
    train_with_forward_forward_algorithm(
        model_type="recurrent",
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--theta", type=float, default=2.)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_memory_profile", type=str, default=None)
    args = parser.parse_args()
    train(**vars(args))