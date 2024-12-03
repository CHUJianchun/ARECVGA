from torch.utils.data import Dataset
import torch
from args import args

device = args.device
dtype = torch.float32


class ProcessedDataset(Dataset):
    def __init__(self, data, tensor=True, repeat=0):
        self.data = {key: val for key, val in data.items()}
        for k in self.data.keys():
            if k != 'smiles_list' and tensor:
                self.data[k] = torch.tensor(self.data[k]).to(dtype)
            if repeat > 1:
                repeat_list = torch.ones(len(self.data[k].shape)).to(torch.int16).tolist()
                repeat_list[0] = repeat
                self.data[k] = self.data[k].repeat(repeat_list)

    def __len__(self):
        return len(self.data['num_atom_list'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}
