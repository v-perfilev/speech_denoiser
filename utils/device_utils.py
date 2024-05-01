import torch
from torch.utils.data import default_collate


def device_collate_fn(batch, use_cuda=False, use_mps=False):
    # Determine the device based on availability and user preference
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Collate the batch using the default collate function from PyTorch
    batch = default_collate(batch)

    # Move each tensor in the batch to the selected device and convert to float32
    batch = [x.to(device).to(torch.float32) for x in batch]

    # Return the processed batch
    return batch


def to_device_fn(obj, use_cuda=False, use_mps=False):
    # Determine the device based on availability and user preference
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Move the object to the selected device and to float32 data type
    return obj.to(device).to(torch.float32)


def to_cpu_fn(obj, use_mps=False):
    if use_mps and obj.device.type == "mps":
        # Explicitly move to CPU
        return obj.to("cpu")
    else:
        # Return the tensor unchanged
        return obj
