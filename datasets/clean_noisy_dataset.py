from torch.utils.data import Dataset


class CleanNoisyDataset(Dataset):
    spectrograms = []

    def __init__(self):
        pass

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx]
