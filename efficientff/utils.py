from collections import Generator

import torch.utils.data


class ProgressiveTrainingDataset(torch.utils.data.Dataset):
    """Dataset for progressive training.
    """
    def __init__(self, dataset_generator: Generator):
        self.internal_dataset = [
            batch
            for data, sign in dataset_generator
            for batch in zip(data, sign)
        ]

    def __getitem__(self, index):
        return self.internal_dataset[index]

    def __len__(self):
        return len(self.internal_dataset)