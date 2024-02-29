import os

def bznsyp(text_path):
    
    split_signals = ['#1', '#2', '#3', '#4']
    
    text_dict = {}
    
    with open(text_path, 'r', encoding='utf8') as f:
        lines = list(f.readlines())
        lines_len = len(lines)
        for i in range(lines_len // 2):
            tmp = lines[i * 2].rstrip().split()
            wav_name = tmp[0]
            assert wav_name == '%06d'%(i + 1)
            text = ''.join(tmp[1:])
            for split_signal in split_signals:
                text = ''.join(text.split(split_signal))
            text_dict[wav_name] = text
    
    return text_dict
            
            
if __name__ == '__main__':
    text_path = r'E:\TTS\data\BZNSYP\ProsodyLabeling\000001-010000.txt'
    out = bznsyp(text_path)
   
    