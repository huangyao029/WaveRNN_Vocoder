import glob
import pickle
import argparse

import numpy as np

from pathlib import Path
from multiprocessing import Pool, cpu_count

from utils import hparams as hp
from utils.files import get_files
from utils.recipes import bznsyp
from utils.dsp import *
from utils.display import *
from utils.paths import Paths

def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0'%(num))
    return n


def convert_file(path : Path):
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.:
        y /= peak
    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(y, mu = 2 ** hp.bits) if hp.mu_law else float_2_label(y, bits = hp.bits)
    if hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits = 16)
    
    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(path : Path):
    wav_id = path.stem
    m, x = convert_file(path)
    np.save(paths.mel/f'{wav_id}.npy', m, allow_pickle = False)
    np.save(paths.quant/f'{wav_id}.npy', x, allow_pickle = False)
    return wav_id, m.shape[-1]
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Preprocessing for WaveRNN and Tacotron.')
    parser.add_argument('--path', '-p', help = 'directly point to dataset path (overrides hparams.wav_path)')
    parser.add_argument('--text_path', '-tp', default = r'E:\TTS\data\BZNSYP\ProsodyLabeling\000001-010000.txt', help = 'the file of text')
    parser.add_argument('--extension', '-e', metavar = 'EXT', default = '.wav', help = 'file extension to search for in dataset folder')
    parser.add_argument('--num_workers', '-w', metavar = 'N', type = valid_n_workers, default = cpu_count() - 1, help = 'The number of worker threads to use for preprocessing')
    parser.add_argument('--hp_file', metavar = 'FILE', default = 'hparams.py', help = 'The file to use for the hyparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)
    if args.path is None:
        args.path = hp.wav_path
        
    extension = args.extension
    path = args.path


    wav_files = get_files(path, extension)
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"')

    if len(wav_files) == 0:
        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')
    else:
        if not hp.ignore_tts:
            text_dict = bznsyp(args.text_path)
            with open(paths.data/'text_dict.pkl', 'wb') as f:
                pickle.dump(text_dict, f)
        
        n_workers = max(1, args.num_workers)
        
        pool = Pool(processes = n_workers)
        dataset = []
        
        for i, (item_id, length) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
            dataset += [(item_id, length)]
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i} / {len(wav_files)}'
            stream(message)
            
        pool.close()
            
        with open(paths.data/'dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            
        print('\n\nCompleted.\n\n')
        
