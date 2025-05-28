import os
import random
import shutil
import string
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from audiomentations import (
    SomeOf,
    AddGaussianNoise,
    BandPassFilter,
    Gain,
    TimeStretch,
    PitchShift,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ID_LENGTH = 20
TARGET_SR = 16000
AUGMENT_PER_FILE = 3
AUGMENT_RATIO = 0.5
MAX_WORKERS = 16

BASE_SPLIT_DIR = 'ATC_ASR_Dataset_Splits'
TRAIN_DIR = os.path.join(BASE_SPLIT_DIR, 'train')
INPUT_AUDIO_DIR = os.path.join(TRAIN_DIR, 'audios')
INPUT_TEXT_DIR = os.path.join(TRAIN_DIR, 'texts')

TEMP_TRAIN_DIR = os.path.join(BASE_SPLIT_DIR, 'train_augmented')
OUTPUT_AUDIO_DIR = os.path.join(TEMP_TRAIN_DIR, 'audios')
OUTPUT_TEXT_DIR = os.path.join(TEMP_TRAIN_DIR, 'texts')

os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

augmenter = SomeOf(
    (2, 3),
    [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.003, p=1.0),
        BandPassFilter(min_center_freq=400.0, max_center_freq=3000.0, p=1.0),
        Gain(min_gain_db=-3.0, max_gain_db=3.0, p=1.0),
        TimeStretch(min_rate=0.97, max_rate=1.03, p=0.5),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
    ],
    p=1.0,
)

def generate_id(length=ID_LENGTH):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

audio_files = [f for f in os.listdir(INPUT_AUDIO_DIR) if f.endswith('.wav')]
random.shuffle(audio_files)

num_to_augment = int(len(audio_files) * AUGMENT_RATIO)
files_to_augment = set(audio_files[:num_to_augment])
total_steps = len(audio_files) + num_to_augment * AUGMENT_PER_FILE

lock = Lock()
progress = tqdm(total=total_steps, desc='Augmenting training split')

def process_file(fname):
    audio_path = os.path.join(INPUT_AUDIO_DIR, fname)
    text_path = os.path.join(INPUT_TEXT_DIR, fname.replace('.wav', '.txt'))
    audio, _ = librosa.load(audio_path, sr=TARGET_SR)
    transcript = open(text_path, encoding='utf-8').read().strip()

    uid = generate_id()
    sf.write(os.path.join(OUTPUT_AUDIO_DIR, f'{uid}.wav'), audio, TARGET_SR)
    open(os.path.join(OUTPUT_TEXT_DIR, f'{uid}.txt'), 'w', encoding='utf-8').write(transcript)

    with lock:
        progress.update(1)

    if fname in files_to_augment:
        for _ in range(AUGMENT_PER_FILE):
            aug_audio = augmenter(samples=audio, sample_rate=TARGET_SR)
            aug_id = generate_id()
            sf.write(os.path.join(OUTPUT_AUDIO_DIR, f'{aug_id}.wav'), aug_audio, TARGET_SR)
            open(os.path.join(OUTPUT_TEXT_DIR, f'{aug_id}.txt'), 'w', encoding='utf-8').write(transcript)
            with lock:
                progress.update(1)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = [ex.submit(process_file, f) for f in audio_files]
    for _ in as_completed(futures):
        pass

progress.close()
print('Augmentation finished; replacing original training split.')

shutil.rmtree(TRAIN_DIR)
os.rename(TEMP_TRAIN_DIR, TRAIN_DIR)

print('Training split updated with augmented data.')
