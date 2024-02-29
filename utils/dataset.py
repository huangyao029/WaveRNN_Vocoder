import torch
import pickle
import random

import numpy as np

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from utils.dsp import *

from utils import hparams as hp
#from hparams import *


class VocoderDataset(Dataset):
    
    def __init__(self, path : Path, dataset_ids, train_gta = False):
        self.metadata = dataset_ids
        self.mel_path = path/'gta' if train_gta else path/'mel'
        self.quant_path = path/'quant'
        
    def __getitem__(self, index):
        item_id = self.metadata[index]
        m = np.load(self.mel_path/f'{item_id}.npy')
        x = np.load(self.quant_path/f'{item_id}.npy')
        return m, x
    
    def __len__(self):
        return len(self.metadata)
    
    
def get_vocoder_datasets(path : Path, batch_size, train_gta):
    
    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
        
    dataset_ids = [x[0] for x in dataset]
    
    random.seed(1234)
    random.shuffle(dataset_ids)
    
    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]
    
    train_dataset = VocoderDataset(path, train_ids, train_gta)
    test_dataset = VocoderDataset(path, test_ids, train_gta)
    
    train_set = DataLoader(
        train_dataset,
        collate_fn = collate_vocoder,
        batch_size = batch_size,
        num_workers = 2,
        shuffle = True,
        pin_memory = True
    )
    
    test_set = DataLoader(
        test_dataset,
        batch_size = 1,
        num_workers = 1,
        shuffle = False,
        pin_memory = True
    )
    
    return train_set, test_set



def collate_vocoder(batch):
    
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i] : mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
    labels = [x[1][sig_offsets[i] : sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]
    
    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)
    
    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()
    
    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]
    
    bits = 16 if hp.voc_mode == 'MOL' else hp.bits
    
    x = label_2_float(x.float(), bits)
    
    if hp.voc_mode == 'MOL':
        y = label_2_float(y.float(), bits)
        
    return x, y, mels

    
    
if __name__ == '__main__':
    batch = [[
        np.load('/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/WaveRNN/mel/000001.npy'),
        np.load('/mnt/storage/hy_workspace/tmp/tmp4/tmp1/tmp9/tts/Tacotron2/data/WaveRNN/quant/000001.npy')
    ]]
    collate_vocoder(batch)
    
