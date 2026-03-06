import torch
from torch.utils.data import Dataset


class RecDataset(Dataset):
    def __init__(self, samples, pad_id):
        self.samples = samples
        self.pad_id = pad_id

    def __getitem__(self, i):
        seq, next_id = self.samples[i]
        len_seq = sum(1 for x in seq if x != self.pad_id)
        if len_seq == 0:
            len_seq = 1
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(len_seq, dtype=torch.long),
            torch.tensor(next_id, dtype=torch.long),
        )

    def __len__(self):
        return len(self.samples)
