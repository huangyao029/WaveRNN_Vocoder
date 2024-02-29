import torch
import argparse
import time
import os

import numpy as np
import torch.nn.functional as F

from torch import optim

from utils.paths import Paths
from utils.dataset import get_vocoder_datasets
from utils.checkpoints import restore_checkpoint
from utils.distribution import discretized_mix_logistic_loss
from utils.display import stream
from utils.dsp import *
from utils import hparams as hp
from utils.checkpoints import save_checkpoint, restore_checkpoint
from utils import data_parallel_workaround
from model import WaveRNN
from gen_wavernn import gen_testset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    
    parser = argparse.ArgumentParser(description = 'Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type = float, help = '[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type = int, help = '[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action = 'store_true', help = 'Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action = 'store_true', help = 'train wavernn on GTA features')
    parser.add_argument('--force_cpu', '-c', action = 'store_true', help = 'Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar = 'FILE', default = 'hparams.py', help = 'The file to use for hyperparameters')
    args = parser.parse_args()
    
    hp.configure(args.hp_file)
    if args.lr is None:
        args.lr = hp.voc_lr
    if args.batch_size is None:
        args.batch_size = hp.voc_batch_size
        
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    
    batch_size = args.batch_size
    force_train = args.force_train
    train_gta = args.gta
    lr = args.lr
    
    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if batch_size % torch.cuda.device_count() != 0:
            raise ValueError('"batch size" must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    
    print('\nInitialising Model...\n')
    
    voc_model = WaveRNN(
        rnn_dims = hp.voc_rnn_dims,
        fc_dims = hp.voc_fc_dims,
        bits = hp.bits,
        pad = hp.voc_pad,
        upsample_factors = hp.voc_upsample_factors,
        feat_dims = hp.num_mels,
        compute_dims = hp.voc_compute_dims,
        res_out_dims = hp.voc_res_out_dims,
        res_blocks = hp.voc_res_blocks,
        hop_length = hp.hop_length,
        sample_rate = hp.sample_rate,
        mode = hp.voc_mode
    ).to(device)
    
    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length
    
    optimizer = optim.Adam(voc_model.parameters())
    restore_checkpoint('voc', paths, voc_model, optimizer, create_if_missing = True)
    
    train_set, test_set = get_vocoder_datasets(paths.data, batch_size, train_gta)
    
    total_steps = 10_000_000 if force_train else hp.voc_total_steps
    
    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss
    
    voc_train_loop(paths, voc_model, loss_func, optimizer, train_set, test_set, lr, total_steps, hp)
    
    
def voc_train_loop(paths : Paths, model : WaveRNN, loss_func, optimizer, train_set, test_set, lr, total_steps, hp):
    
    device = next(model.parameters()).device
    
    for g in optimizer.param_groups:
        g['lr'] = lr
        
    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1
    
    for e in range(1, epochs + 1):
        
        start = time.time()
        running_loss = 0.
        
        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)
            
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                y_hat = data_parallel_workaround(model, x, m)
            else:
                y_hat = model(x, m)
            
            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
                
            y = y.unsqueeze(-1)
            
            loss = loss_func(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                if np.isnan(grad_norm.to('cpu').numpy()):
                    print('grad_norm wav NaN')
            optimizer.step()
            
            running_loss += loss.item()
            avg_loss = running_loss / i
            
            speed = i / (time.time() - start)
            
            step = model.get_step()
            k = step // 1000
            
            if step % hp.voc_checkpoint_every == 0:
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output, hp)
                ckpt_name = f'wave_step{k}K'
                save_checkpoint('voc', paths, model, optimizer, name = ckpt_name, is_silent = True)
                
            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | {speed: .1f} steps/s | Step: {k}k | '
            stream(msg)
            
        save_checkpoint('voc', paths, model, optimizer, is_silent = True)
        model.log(paths.voc_log, msg)
        print(' ')
        
        
if __name__ == '__main__':
    main()
                    
