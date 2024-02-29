import os

from pathlib import Path


class Paths:
    
    def __init__(self, data_path, voc_id, tts_id):
        
        self.base = Path(__file__).parent.parent.expanduser().resolve()
        
        # Data Paths
        self.data = Path(data_path).expanduser().resolve()
        self.quant = self.data/'quant'
        self.mel = self.data/'mel'
        self.gta = self.data/'gta'
        
        # WaveRNN/Vocoder Paths
        self.voc_checkpoints = self.base/'checkpoints'/f'{voc_id}.wavernn'
        self.voc_latest_weights = self.voc_checkpoints/'latest_weights.pyt'
        self.voc_latest_optim = self.voc_checkpoints/'latest_optim.pyt'
        self.voc_output = self.base/'model_outputs'/f'{voc_id}.wavernn'
        self.voc_step = self.voc_checkpoints/'step.npy'
        self.voc_log = self.voc_checkpoints/'log.txt'
        
        self.create_paths()
        
    def create_paths(self):
        os.makedirs(self.data, exist_ok = True)
        os.makedirs(self.quant, exist_ok = True)
        os.makedirs(self.mel, exist_ok = True)
        os.makedirs(self.gta, exist_ok = True)
        os.makedirs(self.voc_checkpoints, exist_ok = True)
        os.makedirs(self.voc_output, exist_ok = True)
        
    def get_voc_named_weights(self, name):
        return self.voc_checkpoints/f'{name}_weights.pyt'
    
    def get_voc_named_optim(self, name):
        return self.voc_checkpoints/f'{name}_optim.pyt'
        
        
if __name__ == '__main__':
    obj = Paths(r'E:\TTS\data\BZNSYP\WaveRNN', 1, 1)