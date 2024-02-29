# CONFIG

wav_path = '/ZFS4T/tts/data/Wave/'
data_path = '/ZFS4T/tts/data/WaveRNN/'

voc_model_id = 'bznsyp_mol'
tts_model_id = 'bznsyp_lsa_smooth_attention'

ignore_tts = True


# DSP
sample_rate = 48000
n_fft = 4096
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = int(sample_rate * 0.0125)
win_length = int(sample_rate * 0.05)
fmin = 0
fmax = sample_rate // 2
min_level_db = -100
ref_level_db = 20
bits = 9
mu_law = True
peak_norm = False


# WAVERNN / VOCODER

# Model Hparams
voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
                                    # 'MOL' means 'Mixture Of Logistic'
voc_upsample_factors = (5, 8, 15)   # NB - this needs to correctly factories hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 48
#voc_lr = 1e-4
voc_lr = 0.000002
voc_checkpoint_every = 2_000
voc_gen_at_checkpoint = 5
voc_total_steps = 1_000_000
voc_test_samples = 50
voc_pad = 2
voc_seq_len = hop_length * 5
voc_clip_grad_norm = 4

# Gnerating / Synthesizing
voc_gen_batched = False
voc_target = 11_000
voc_overlap = 550