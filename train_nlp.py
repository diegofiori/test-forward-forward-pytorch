import argparse
import urllib.request

import torch
import torch.utils.data

from efficientff.modules import LMFFNet
from efficientff.utils import compute_perplexity


def download_fables():
    http_str = "http://classics.mit.edu/Aesop/fab.mb.txt"
    with urllib.request.urlopen(http_str) as response:
        html = response.read()
    return html.decode("utf-8")


def get_fables():
    fables = download_fables()
    fables = fables.split("SECTION 1")[1]
    fables = fables.split("THE END")[0]
    fables = fables.split("\n\n")
    fables = [fable for fable in fables if len(fable) >= 100]
    return fables


vocabulary = {
    " ": 0,
    "!": 1,
    ",": 2,
    ".": 3,
    "a": 4,
    "b": 5,
    "c": 6,
    "d": 7,
    "e": 8,
    "f": 9,
    "g": 10,
    "h": 11,
    "i": 12,
    "j": 13,
    "k": 14,
    "l": 15,
    "m": 16,
    "n": 17,
    "o": 18,
    "p": 19,
    "q": 20,
    "r": 21,
    "s": 22,
    "t": 23,
    "u": 24,
    "v": 25,
    "w": 26,
    "x": 27,
    "y": 28,
    "z": 29,
}


def tokenize(fable, max_len=100):
    tokenized_fable = [vocabulary[char] for i, char in enumerate(fable.lower()) if char in vocabulary]
    return tokenized_fable[:max_len]


def get_tokenized_fables():
    fables = get_fables()
    tokenized_fables = [tokenize(fable) for fable in fables]
    tokenized_fables = torch.stack([torch.tensor(tokens) for tokens in tokenized_fables if len(tokens) == 100])
    return tokenized_fables


def get_dataloader(batch_size=32, test_size=0.2):
    tokenized_fables = get_tokenized_fables()
    n_test = int(len(tokenized_fables) * test_size)
    test_set = torch.utils.data.TensorDataset(tokenized_fables[:n_test])
    train_set = torch.utils.data.TensorDataset(tokenized_fables[n_test:])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=n_test, shuffle=True)
    return train_loader, test_loader
    

def train(
        n_layers: int, 
        hidden_size: int, 
        sequence_len: int, 
        epochs: int, 
        batch_size: int, 
        loss_fn: str, 
        lr: float, 
        theta: float, 
        device: str
):
    train_loader, test_loader = get_dataloader(batch_size=batch_size)
    token_num = len(vocabulary)
    optimizer_name = "Adam"
    optimizer_kwargs = {"lr": lr}
    model = LMFFNet(
        n_layers=n_layers, 
        hidden_size=hidden_size, 
        token_num=token_num, 
        seq_len=sequence_len, 
        predicted_tokens=100-sequence_len,
        epochs=epochs,
        loss_fn_name=loss_fn, 
        optimizer_name=optimizer_name, 
        optimizer_kwargs=optimizer_kwargs
    ).to(device)
    for input_data in train_loader:
        input_data = torch.functional.F.one_hot(input_data[0].to(device), num_classes=token_num).float()
        print(input_data.shape)
        accumulated_goodness = model.LM_ff_train(input_data, theta=theta)
        print("Trained on batch")
        print(f"Accumulated goodness: {accumulated_goodness}")
        print(f"Accumulated goodness ratio: {(accumulated_goodness[0]-accumulated_goodness[1]) / abs(max(accumulated_goodness))}")
    
    for test_data in test_loader:
        test_data = torch.functional.F.one_hot(test_data[0].to(device), num_classes=token_num).float()
        test_data = test_data.reshape(-1, token_num*sequence_len)
        predictions, _ = model.positive_eval(test_data, theta)
        perplexity = compute_perplexity(predictions)
        print(f"Perplexity: {perplexity}")
        # predicted_tokens = torch.argmax(predictions, dim=2)
        # inverse_vocabulary = {v: k for k, v in vocabulary.items()}
        # for line in predicted_tokens:
        #     print("".join([inverse_vocabulary[token.item()] for token in line]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--sequence_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss_fn", type=str, default="alternative_loss_fn")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--theta", type=float, default=2.)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(**vars(args))