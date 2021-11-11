from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Hparams

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def spk2indices(utt_ids):
    hps = Hparams
    spk_table = hps.SPEAKERS.spk_to_inds
    spk_names = [utt_id.split('_')[0] for utt_id in utt_ids]
    spk_ids = [spk_table.index(spk) for spk in spk_names]
    return spk_ids


def collate_fn(data):
    """
       data: is a list of dictionary with (fid, bnf, mel, linear, length)
             where 'bnf, mel, linear,' are tensors of arbitrary lengths
             and fid/length are scalars
    """
    fids = [d['fid'] for d in data]
    bnfs = [d['bnf'] for d in data]
    mels = [d['mel'] for d in data]
    linears = [d['linear'] for d in data]
    f0s = [d['f0'] for d in data]
    lengths = [d['length'] for d in data]
    bnfs_batch = [torch.Tensor(bnf) for bnf in bnfs]
    mels_batch = [torch.Tensor(mel) for mel in mels]
    linears_batch = [torch.Tensor(linear) for linear in linears]
    f0s_batch = [torch.Tensor(lf0) for lf0 in f0s]
    lengths_batch = torch.Tensor(lengths)
    spkids_batch = torch.LongTensor(spk2indices(fids))
    bnfs_batch = torch.nn.utils.rnn.pad_sequence(bnfs_batch)
    mels_batch = torch.nn.utils.rnn.pad_sequence(mels_batch)
    linears_batch = torch.nn.utils.rnn.pad_sequence(linears_batch)
    f0s_batch = torch.nn.utils.rnn.pad_sequence(f0s_batch)

    return {'fid': fids,
            'bnf': bnfs_batch.float(),
            'mel': mels_batch.float(),
            'linear': linears_batch.float(),
            'length': lengths_batch.int(),
            'f0': f0s_batch.float(),
            'spkid': spkids_batch}


class VCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.data_dir = data_dir
        self.data_dict = self._get_paths(csv_file)
        self.transform = transform

    def _get_paths(self, csv_file):
        fids = []
        bnf_paths = []
        mel_paths = []
        linear_paths = []
        f0_paths = []
        with open(csv_file, 'r') as f:
            for line in f:
                line = line.strip()
                lst = line.split('|')
                fids.append(lst[0])
                bnf_paths.append(os.path.join(self.data_dir, lst[1]))
                mel_paths.append(os.path.join(self.data_dir, lst[2]))
                linear_paths.append(os.path.join(self.data_dir, lst[3]))
                f0_paths.append(os.path.join(self.data_dir, lst[4]))
        return {'fids': fids,
                'bnfs': bnf_paths,
                'mels': mel_paths,
                'linears': linear_paths,
                'f0s': f0_paths}

    def __len__(self):
        return len(self.data_dict['fids'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fid = self.data_dict['fids'][idx]
        bnf = np.load(self.data_dict['bnfs'][idx])
        mel = np.load(self.data_dict['mels'][idx])
        linear = np.load(self.data_dict['linears'][idx])
        try:
            f0 = np.load(self.data_dict['f0s'][idx])
        except ValueError as e:
            print(self.data_dict['f0s'][idx])
        sample = {'fid': fid, 'bnf': bnf, 'mel': mel,
                  'linear': linear, 'f0': f0,
                  'length': bnf.shape[0]}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    # dataloader = DataLoader()
    data_set = VCDataset('/datapool/home/lu-h17/datasets/cmu-arctic-slt/test_meta.csv')
    dataloader = DataLoader(data_set, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for idx, batch in enumerate(dataloader):
        print(idx, batch['bnf'].shape, batch['mel'].shape, batch['f0'].shape, batch['length'])
        inputs = torch.cat([batch['bnf'], batch['f0']], dim=2)
        print(inputs.shape)
