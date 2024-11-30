import pickle

import lmdb
from torch.utils.data import Dataset


class LmdbCleanNoisyDataset(Dataset):
  """Dataset for loading saved spectrograms."""

  def __init__(self, lmdb_path):
    """Initialize the dataset with directories containing clean and noisy spectrograms."""
    self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with self.env.begin(write=False) as txn:
      self.length = txn.stat()["entries"]

  def __len__(self):
    """Return the number of spectrogram pairs in the dataset."""
    return self.length

  def __getitem__(self, idx):
    """Retrieve a spectrogram pair by index."""
    key = f"spectrogram_{idx:06d}".encode("ascii")
    with self.env.begin(write=False) as txn:
      data_pair = pickle.loads(txn.get(key))
    return data_pair["noisy"], data_pair["clean"]
