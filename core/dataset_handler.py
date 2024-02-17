from torch.utils.data import DataLoader, random_split


class DatasetHandler:
    batch_size = 16

    def split_dataset_into_data_loaders(self, dataset, count=None, train_ratio=0.8):
        if count is not None and count < len(dataset):
            dataset, _ = random_split(dataset, [count, len(dataset) - count])

        dataset_size = len(dataset)
        train_size = round((dataset_size * train_ratio) / self.batch_size) * self.batch_size

        train_dataset, val_dataset = random_split(dataset, [train_size, dataset_size - train_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
