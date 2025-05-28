import os
import random
import uuid
import re
import xml.etree.ElementTree as ET
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils import atco2_general_corrections

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INPUT_DIR = 'ATCO2_Raw_Data'
DATASET_DIR = 'ATCO2_Dataset'
AUDIO_OUTPUT_DIR = os.path.join(DATASET_DIR, 'audios')
TEXT_OUTPUT_DIR = os.path.join(DATASET_DIR, 'texts')

os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

TAGS_REMOVE = [
    r'\[#command\]', r'\[/#command\]', r'\[#value\]', r'\[/#value\]',
    r'\[#unnamed\]', r'\[/#unnamed\]', r'\[#callsign\]', r'\[/#callsign\]',
    r'\[hes\]', r'\[HES\]', r'\[noise\]',
]

EXCLUDE_IF_CONTAINS = [
    r'\[NE Czech\]', r'\[/NE\]', r'\[NE Slovak\]', r'\[Ne Czech\]',
    r'\[/Ne\]', r'\[Ne Slovak\]', r'\[#nonenglish\]', r'\[/#nonenglish\]',
    r'\[ukn\]', r'\[spk\]', r'\[xt\]', r'[\(\)]',
]


def deterministic_uuid():
    return uuid.UUID(int=random.getrandbits(128))


def clean_transcript(text):
    if any(re.search(tag, text, re.IGNORECASE) for tag in EXCLUDE_IF_CONTAINS):
        return None
    for tag in TAGS_REMOVE:
        text = re.sub(tag, '', text, flags=re.IGNORECASE)
    for wrong, correct in atco2_general_corrections.items():
        pattern = re.compile(rf'\b{re.escape(wrong)}\b', re.IGNORECASE)
        text = pattern.sub(correct, text)
    cleaned = re.sub(r'\s{2,}', ' ', text.strip()).upper()
    return cleaned or None


def process_file(filename):
    xml_path = os.path.join(INPUT_DIR, filename)
    wav_path = os.path.join(INPUT_DIR, filename.replace('.xml', '.wav'))
    if not os.path.exists(wav_path):
        return
    try:
        audio_data, sample_rate = sf.read(wav_path)
        tree = ET.parse(xml_path)
    except Exception:
        return
    root = tree.getroot()
    for segment in root.findall('segment'):
        tags = segment.find('tags')
        if tags is None:
            continue
        if tags.findtext('non_english') != '0' or tags.findtext('correct_transcript') != '1':
            continue
        try:
            start = float(segment.findtext('start'))
            end = float(segment.findtext('end'))
            raw_text = segment.findtext('text', '').strip()
            if not raw_text or end <= start:
                continue
            cleaned_text = clean_transcript(raw_text)
            if not cleaned_text:
                continue
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            uid = deterministic_uuid().hex.upper()[:20]
            sf.write(os.path.join(AUDIO_OUTPUT_DIR, f'{uid}.wav'), segment_audio, sample_rate)
            with open(os.path.join(TEXT_OUTPUT_DIR, f'{uid}.txt'), 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
        except Exception:
            continue


def main():
    xml_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xml')]
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(
            tqdm(
                executor.map(process_file, xml_files),
                total=len(xml_files),
                desc='Processing Dataset',
            )
        )
    print('ATCO2 dataset processing completed.')


if __name__ == '__main__':
    main()
