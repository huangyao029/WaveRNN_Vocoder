import librosa
import math

import numpy as np
import soundfile as sf

from utils import hparams as hp


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.
    x = (x + 1.) * (2 ** bits - 1) / 2
    return x.clip(0, 2 ** bits - 1)


def load_wav(path):
    return librosa.load(path, sr = hp.sample_rate)[0]


def stft(y):
    return librosa.stft(
        y = y,
        n_fft = hp.n_fft,
        hop_length = hp.hop_length,
        win_length = hp.win_length
    )
    
    
def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10., x * 0.05)


def melspectrogram(y):
    fbank = librosa.feature.melspectrogram(
        y = y,
        sr = hp.sample_rate,
        n_fft = hp.n_fft,
        win_length = hp.win_length,
        hop_length = hp.hop_length,
        n_mels = hp.num_mels,
        fmin = hp.fmin,
        fmax = hp.fmax
    )
    log_fbank = librosa.power_to_db(fbank, ref = np.max)
    return log_fbank


def encode_mu_law(x, mu):
    mu -= 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels = True):
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu -= 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def save_wav(x, path):
    #librosa.output.write_wav(path, x.astype(np.float32), sr = hp.sample_rate)
    sf.write(path, x.astype(np.float32), hp.sample_rate)

# if __name__ == '__main__':
#     x = np.array([123, 3232, 42435])
#     mu = 2 ** 9
#     print(encode_mu_law(x, mu))