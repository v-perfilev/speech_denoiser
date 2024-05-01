from torch.utils.data import Dataset


class CvCleanNoisyDataset(Dataset):
    """A mock dataset."""

    def __init__(self):
        self.spectrograms = []

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx]
