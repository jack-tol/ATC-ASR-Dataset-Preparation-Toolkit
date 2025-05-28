import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SOURCE_DIR = Path('ATC_ASR_Dataset')
OUTPUT_DIR = Path('ATC_ASR_Dataset_Splits')
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

AUDIO_DIR = SOURCE_DIR / 'audios'
TEXT_DIR = SOURCE_DIR / 'texts'

uuids = [
    f.stem
    for f in AUDIO_DIR.glob('*.wav')
    if (TEXT_DIR / f.with_suffix('.txt').name).exists()
]
random.shuffle(uuids)

total = len(uuids)
train_end = int(TRAIN_RATIO * total)
val_end = train_end + int(VAL_RATIO * total)

splits = {
    'train': uuids[:train_end],
    'validation': uuids[train_end:val_end],
    'test': uuids[val_end:],
}

def copy_entry(uid, split_name):
    (OUTPUT_DIR / split_name / 'audios').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split_name / 'texts').mkdir(parents=True, exist_ok=True)
    shutil.copy2(AUDIO_DIR / f'{uid}.wav', OUTPUT_DIR / split_name / 'audios' / f'{uid}.wav')
    shutil.copy2(TEXT_DIR / f'{uid}.txt', OUTPUT_DIR / split_name / 'texts' / f'{uid}.txt')

with ThreadPoolExecutor() as ex:
    futures = [
        ex.submit(copy_entry, uid, split)
        for split, lst in splits.items()
        for uid in lst
    ]
    for _ in tqdm(as_completed(futures), total=len(futures), desc='Splitting Dataset'):
        pass

print('Dataset split completed.')