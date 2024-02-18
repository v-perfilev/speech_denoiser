import torch
from torch.utils.data import DataLoader, random_split


class DatasetHandler:

    def __init__(self, batch_size=32, use_mps=False):
        self.batch_size = batch_size
        self.use_mps = use_mps

    def split_dataset_into_data_loaders(self, dataset, count=None, train_ratio=0.8):
        if count is not None and count < len(dataset):
            dataset, _ = random_split(dataset, [count, len(dataset) - count])

        dataset_size = len(dataset)
        train_size = round((dataset_size * train_ratio) / self.batch_size) * self.batch_size

        train_dataset, val_dataset = random_split(dataset, [train_size, dataset_size - train_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        return train_loader, val_loader

    def collate_fn(self, batch):
        device = torch.device("mps" if self.use_mps and torch.backends.mps.is_available() else "cpu")
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = [x.to(device).to(torch.float32) for x in batch]
        return batch
