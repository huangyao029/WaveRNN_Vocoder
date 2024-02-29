from pathlib import Path
from typing import Union

def get_files(path, extension = '.wav'):
    if isinstance(path, str):
        path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

if __name__ == '__main__':
    out = get_files(r'E:\TTS\data\BZNSYP\Wave')