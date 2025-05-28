import os
import sys
import shutil
import soundfile as sf
import resampy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TARGET_SR = 16000
DEFAULT_DATASETS = ['ATCC_Dataset', 'ATCO2_Dataset', 'UWB_Dataset']
DESTINATION = 'ATC_ASR_Dataset'
DEST_AUDIO_DIR = os.path.join(DESTINATION, 'audios')
DEST_TEXT_DIR = os.path.join(DESTINATION, 'texts')

os.makedirs(DEST_AUDIO_DIR, exist_ok=True)
os.makedirs(DEST_TEXT_DIR, exist_ok=True)

datasets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DATASETS
pairs = []

for ds in datasets:
    audio_dir = os.path.join(ds, 'audios')
    text_dir = os.path.join(ds, 'texts')
    if not (os.path.isdir(audio_dir) and os.path.isdir(text_dir)):
        print(f'Skipping missing dataset: {ds}')
        continue
    audios = {os.path.splitext(f)[0]: os.path.join(audio_dir, f)
              for f in os.listdir(audio_dir) if f.endswith('.wav')}
    texts = {os.path.splitext(f)[0]: os.path.join(text_dir, f)
             for f in os.listdir(text_dir) if f.endswith('.txt')}
    for key in set(audios) & set(texts):
        pairs.append((key, audios[key], texts[key]))

def process_entry(key, audio_path, text_path):
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != TARGET_SR:
            audio = resampy.resample(audio, sr, TARGET_SR)
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(os.path.join(DEST_AUDIO_DIR, f'{key}.wav'), audio, TARGET_SR, subtype='PCM_16')
        shutil.copy2(text_path, os.path.join(DEST_TEXT_DIR, f'{key}.txt'))
    except Exception:
        pass

with ThreadPoolExecutor() as ex:
    futures = [ex.submit(process_entry, k, a, t) for k, a, t in pairs]
    for _ in tqdm(as_completed(futures), total=len(futures), desc='Creating ATC_ASR_Dataset'):
        pass

print('ATC_ASR_Dataset processing completed.')