import os
import random
import string
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from tqdm import tqdm
from utils import atc_0_general_corrections

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SEGMENT_ID_LENGTH = 20

INPUT_DIR = 'ATCC_Raw_Data'
DATASET_DIR = 'ATCC_Dataset'
AUDIO_OUTPUT_DIR = os.path.join(DATASET_DIR, 'audios')
TEXT_OUTPUT_DIR = os.path.join(DATASET_DIR, 'texts')
SUBFOLDERS = ['atc0_bos', 'atc0_dca', 'atc0_dfw']

for d in (AUDIO_OUTPUT_DIR, TEXT_OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)


def normalize_audio_files(folder_path):
    audio_path = os.path.join(folder_path, 'data', 'audio')
    for fname in os.listdir(audio_path):
        if fname.endswith(('.sph', '.wav')):
            in_path = os.path.join(audio_path, fname)
            base, _ = os.path.splitext(in_path)
            tmp_path = base + '.temp.wav'
            out_path = base + '.wav'
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', in_path, '-ar', '16000', '-ac', '1', tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                os.replace(tmp_path, out_path)
                if fname.endswith('.sph'):
                    os.remove(in_path)
            except subprocess.CalledProcessError:
                continue


def generate_unique_id(existing_ids, length=SEGMENT_ID_LENGTH):
    chars = string.ascii_uppercase + string.digits
    while True:
        uid = ''.join(random.choices(chars, k=length))
        if uid not in existing_ids:
            existing_ids.add(uid)
            return uid


def clean_whitespace(s):
    return re.sub(r'\s+', ' ', s).strip()


def extract_balanced_parens(data, start_idx):
    depth = 0
    for idx in range(start_idx, len(data)):
        if data[idx] == '(':
            depth += 1
        elif data[idx] == ')':
            depth -= 1
            if depth == 0:
                return data[start_idx + 1:idx], idx + 1
    raise ValueError


def parse_transcript(path):
    data = open(path, encoding='utf-8').read()
    idx = 0
    out = []
    while True:
        tpos = data.find('(TEXT', idx)
        if tpos < 0:
            break
        raw_text, after_text = extract_balanced_parens(data, tpos)
        parts = raw_text.split(maxsplit=1)
        text = clean_whitespace(parts[1] if len(parts) == 2 else '')
        t2 = data.find('(TIMES', after_text)
        if t2 < 0:
            break
        raw_times, idx = extract_balanced_parens(data, t2)
        times = raw_times.split()
        if len(times) >= 3:
            start, end = float(times[1]), float(times[2])
            out.append((text, start, end))
    return out


TAGS_OMIT = [
    '(UNINTELLIGIBLE)',
    'UNINTELLIGIBLE',
    '(MIC KEYED TWICE)',
    '(CARRIER TRANSMITTED ONLY)',
]

TAGS_REMOVE = [
    'LONG PAUSE',
    'SHORT PAUSE',
    'BREAK',
    'GARBLED',
    'LAUGHTER',
    '/',
    '//',
]

OMIT_REGEX = [
    re.compile(r'\(?'+t.replace(' ', '[-\\s]')+r'\)?', re.IGNORECASE) for t in TAGS_REMOVE
]
STUTTER_RE = re.compile(r'\b\w+-\s+(\w+)\b', re.IGNORECASE)
QUOTE_RE = re.compile(r'\(QUOTE\s+([A-Z]+)\)', re.IGNORECASE)
PLAIN_QUOTE = re.compile(r'\bQUOTE\s+([A-Z]+)\b', re.IGNORECASE)
CONTRACTION_RE = re.compile(
    r"\b([A-Z]+)\s+'\s*(RE|LL|VE|D|S|M|T|AM)\b", re.IGNORECASE
)
OCLOCK_RE = re.compile(r"\b(O)\s+'\s*(CLOCK)\b", re.IGNORECASE)


def fix_quotes(line):
    line = QUOTE_RE.sub(lambda m: f"'{m.group(1)}", line)

    def repl(m):
        return f"'{m.group(1)}" if m.start() == 0 or line[m.start() - 1].isspace() else m.group(0)

    return PLAIN_QUOTE.sub(repl, line)


def process_segment(audio, raw_text, start_s, end_s, used_ids):
    if any(tag in raw_text for tag in TAGS_OMIT):
        return None
    if re.search(r'\d', raw_text) or '[' in raw_text or ']' in raw_text:
        return None
    txt = raw_text.replace(';', '').replace('`', "'")
    for rx in OMIT_REGEX:
        txt = rx.sub('', txt)
    txt = txt.replace('(', ' ').replace(')', ' ')
    if 'QUOTE' in txt or '(QUOTE' in txt:
        txt = fix_quotes(txt)
    txt = STUTTER_RE.sub(r'\1', txt)
    txt = re.sub(r'\s*-\s*$', '', txt)
    txt = CONTRACTION_RE.sub(r"\1'\2", txt)
    txt = OCLOCK_RE.sub(r"\1'\2", txt)
    for wrong, right in atc_0_general_corrections.items():
        txt = re.sub(rf'\b{re.escape(wrong)}\b', right, txt, flags=re.IGNORECASE)
    txt = clean_whitespace(txt)
    if not txt or '"' in txt:
        return None
    uid = generate_unique_id(used_ids)
    clip = audio[int(start_s * 1000):int(end_s * 1000)].set_frame_rate(16000)
    clip.export(os.path.join(AUDIO_OUTPUT_DIR, f'{uid}.wav'), format='wav')
    with open(os.path.join(TEXT_OUTPUT_DIR, f'{uid}.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    return txt


def main():
    used_ids = set()
    futures = []
    with ThreadPoolExecutor() as executor:
        for folder in SUBFOLDERS:
            fp = os.path.join(INPUT_DIR, folder)
            normalize_audio_files(fp)
            audio_dir = os.path.join(fp, 'data', 'audio')
            transcript_dir = os.path.join(fp, 'data', 'transcripts')
            if not os.path.isdir(audio_dir) or not os.path.isdir(transcript_dir):
                continue
            wavs = {
                os.path.splitext(f)[0]: os.path.join(audio_dir, f)
                for f in os.listdir(audio_dir)
                if f.endswith('.wav')
            }
            txts = {
                os.path.splitext(f)[0]: os.path.join(transcript_dir, f)
                for f in os.listdir(transcript_dir)
                if f.endswith('.txt')
            }
            for k in set(wavs) & set(txts):
                audio = AudioSegment.from_wav(wavs[k])
                for raw, s, e in parse_transcript(txts[k]):
                    futures.append(
                        executor.submit(
                            process_segment, audio, raw, s, e, used_ids
                        )
                    )
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Processing Dataset',
        ):
            pass
    print('ATCC dataset processing completed.')


if __name__ == '__main__':
    main()
