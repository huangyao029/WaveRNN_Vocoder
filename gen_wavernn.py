
from pathlib import Path

from model import WaveRNN
from utils.dsp import *


def gen_testset(model : WaveRNN, test_set, samples, batched, target, overlap, save_path : Path, hp):
    
    k = model.get_step() // 1000
    
    for i, (m, x) in enumerate(test_set, 1):
        
        if i > samples:
            break
        
        print('\n| Generating : %i/%i'%(i, samples))
        
        x = x[0].numpy()
        
        bits = 16 if hp.voc_mode == 'MOL' else hp.bits
        
        if hp.mu_law and hp.voc_mode != 'MOL':
            x = decode_mu_law(x, 2 ** bits, from_labels = True)
        else:
            x = label_2_float(x, bits)
            
        save_wav(x, save_path/f'{k}k_step_{i}_target.wav')
        
        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = str(save_path/f'{k}k_steps_{i}_{batch_str}.wav')
        
        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)
            
        