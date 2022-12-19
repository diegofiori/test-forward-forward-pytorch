import json
import subprocess
from pathlib import Path

# from pytorch_memlab import LineProfiler, MemReporter


def profile_per_num_layers(n_layers: int):
    hidden_size = 2000
    lr = 0.03
    # device = "cuda:0"
    device = "cpu"
    epochs = 1
    batch_size = 5000
    theta = 2.0
    # reporter_backprop = MemReporter(run_backprop_mnist)
    profiled_values = {}

    CMD = [
        "python",
        "train_base.py",
        "--n_layers",
        str(n_layers),
        "--hidden_size",
        str(hidden_size),
        "--lr",
        str(lr),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--theta",
        str(theta),
        "--save_memory_profile",
        f"base_ff.txt",
    ]
    subprocess.run(CMD)
    with open(f"base_ff.txt", "r") as f:
        string_profile = f.read()
        profiled_values["base_ff"] = float(string_profile.strip())
    Path(f"base_ff.txt").unlink()

    CMD = [
        "python",
        "train_recurrent.py",
        "--n_layers",
        str(n_layers),
        "--hidden_size",
        str(hidden_size),
        "--lr",
        str(lr),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--theta",
        str(theta),
        "--save_memory_profile",
        f"recurrent_ff.txt",
    ]
    subprocess.run(CMD)
    with open("recurrent_ff.txt", "r") as f:
        string_profile = f.read()
        profiled_values["recurrent_ff"] = float(string_profile.strip())
    Path(f"recurrent_ff.txt").unlink()

    CMD = [
        "python",
        "train_backprop.py",
        "--n_layers",
        str(n_layers),
        "--hidden_size",
        str(hidden_size),
        "--lr",
        str(lr),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--save_memory_profile",
        f"backprop.txt",
    ]
    subprocess.run(CMD)
    with open(f"backprop.txt", "r") as f:
        string_profile = f.read()
        profiled_values["backprop"] = float(string_profile.strip())
    Path(f"backprop.txt").unlink()

    print("##############################################")
    print("RESULTS ON MEMORY CONSUMPTION")
    print(profiled_values)
    return profiled_values


def main(max_layers: int):
    all_results = {}
    for n_layers in range(2, max_layers + 1, 5):
        profile_result = profile_per_num_layers(n_layers)
        all_results[n_layers] = profile_result
    with open("results.json", "w") as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_layers", type=int, default=10)
    args = parser.parse_args()
    main(args.max_layers)
