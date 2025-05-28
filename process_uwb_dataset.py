import os
import re
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from tqdm import tqdm
from utils import (
    uwb_general_corrections,
    uwb_diacritics,
    uwb_transmissions_to_specifically_exclude,
    uwb_phonetic_mapping,
    uwb_number_mapping,
    uwb_tags_to_remove,
    uwb_exclude_if_contains,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INPUT_DIR = 'UWB_Raw_Data'
DATASET_DIR = 'UWB_Dataset'
AUDIO_OUTPUT_DIR = os.path.join(DATASET_DIR, 'audios')
TEXT_OUTPUT_DIR = os.path.join(DATASET_DIR, 'texts')

os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

COMPILED_TAGS_REMOVE = [re.compile(p, re.IGNORECASE) for p in uwb_tags_to_remove]
COMPILED_EXCLUSION = [re.compile(p, re.IGNORECASE) for p in uwb_exclude_if_contains]


def generate_uid(length=20):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))


def replace_phonetic(m):
    l = m.group(1).upper()
    return uwb_phonetic_mapping.get(l, l)


def convert_atc_text(text):
    text = re.sub(
        r'(?<=\s)([b-hj-zB-HJ-Z])(?=\s)',
        replace_phonetic,
        f' {text} ',
    ).strip()
    for k, v in uwb_general_corrections.items():
        text = re.sub(rf'\b{re.escape(k)}\b', v, text, flags=re.IGNORECASE)
    tokens = text.split()
    out = []
    i = 0
    nato = set(uwb_phonetic_mapping.values())
    triggers = {
        'CONFIRM',
        'REQUEST',
        'DESCEND',
        'CLIMB',
        'MAINTAIN',
        'CLEARED',
        'CONTACT',
        'REPORT',
        'BACKTRACK',
        'LINE',
        'UP',
        'DOWN',
        'IDENT',
        'SQUAWK',
        'COPY',
        'ROGER',
        'WILCO',
        'ACKNOWLEDGE',
        'DEPART',
        'APPROACH',
    }
    while i < len(tokens):
        t = tokens[i]
        if (
            len(t) == 1
            and t.isalpha()
            and t.isupper()
            and i + 2 < len(tokens)
            and tokens[i + 1].upper() == 'AND'
            and len(tokens[i + 2]) == 1
            and tokens[i + 2].isalpha()
            and tokens[i + 2].isupper()
        ):
            out.extend(
                [
                    uwb_phonetic_mapping[t],
                    'AND',
                    uwb_phonetic_mapping[tokens[i + 2]]
                    if tokens[i + 2] != 'I'
                    else 'I',
                ]
            )
            i += 3
            continue
        if (
            t.upper() == 'AND'
            and i + 1 < len(tokens)
            and len(tokens[i + 1]) == 1
            and tokens[i + 1].isalpha()
            and tokens[i + 1].isupper()
        ):
            out.append('AND')
            out.append(
                uwb_phonetic_mapping[tokens[i + 1]]
                if tokens[i + 1] != 'I'
                else 'I'
            )
            i += 2
            continue
        if t == 'FL':
            out.append('FLIGHT LEVEL')
        elif t in uwb_number_mapping:
            out.append(uwb_number_mapping[t])
        elif len(t) == 1 and t.isalpha() and t.isupper():
            nxt = (
                i < len(tokens) - 1
                and (
                    (
                        len(tokens[i + 1]) == 1
                        and tokens[i + 1].isalpha()
                        and tokens[i + 1].isupper()
                    )
                    or tokens[i + 1].isdigit()
                    or tokens[i + 1] in nato
                    or tokens[i + 1] in triggers
                )
            )
            prev = (
                i > 0
                and (
                    (
                        len(tokens[i - 1]) == 1
                        and tokens[i - 1].isalpha()
                        and tokens[i - 1].isupper()
                    )
                    or tokens[i - 1].isdigit()
                    or tokens[i - 1] in nato
                )
            )
            if i == 0:
                out.append(
                    uwb_phonetic_mapping[t]
                    if nxt
                    else (t if t == 'I' else uwb_phonetic_mapping.get(t, t))
                )
            else:
                out.append(uwb_phonetic_mapping[t] if prev or nxt else t)
        else:
            out.append(t)
        i += 1
    if out and len(out[-1]) == 1 and out[-1].isalpha() and out[-1].isupper():
        out[-1] = uwb_phonetic_mapping[out[-1]]
    return ' '.join(out)


def clean_text(text):
    for d, r in uwb_diacritics.items():
        text = text.replace(d, r)
    for p in COMPILED_EXCLUSION:
        if p.search(text):
            return text, True
    t = text
    for p in COMPILED_TAGS_REMOVE:
        t = p.sub(' ', t)
    t = re.sub(r'\.{2,}', ' ', t)
    t = t.replace('?', '')
    t = re.sub(r'[^\w\s.]', '', t)
    t = re.sub(r'(?<=\d)\.(?=\s|$)', ' .', t)
    t = re.sub(r'\s+', ' ', t).strip().upper()
    return convert_atc_text(t), False


SYNC_PATTERN = re.compile(r'<Sync time="([\d.]+)"/>\s*([^<]*)')

normalized_excluded = {
    clean_text(
        re.sub(
            r'(?<=[0-9])(?=[A-Za-z])',
            ' ',
            re.sub(
                r'(?<=[A-Za-z])(?=[0-9])',
                ' ',
                t,
            ),
        )
    )[0].strip().upper()
    for t in uwb_transmissions_to_specifically_exclude
    if clean_text(
        re.sub(
            r'(?<=[0-9])(?=[A-Za-z])',
            ' ',
            re.sub(
                r'(?<=[A-Za-z])(?=[0-9])',
                ' ',
                t,
            ),
        )
    )[0]
}
strict_exclusions = {
    s.strip().upper() for s in uwb_transmissions_to_specifically_exclude
}


def process_file(filename):
    base = os.path.splitext(filename)[0]
    trs_path = os.path.join(INPUT_DIR, f'{base}.trs')
    wav_path = os.path.join(INPUT_DIR, f'{base}.wav')
    if not os.path.exists(wav_path):
        return []
    try:
        content = open(trs_path, encoding='cp1250').read()
    except Exception:
        return []
    matches = SYNC_PATTERN.findall(content)
    if not matches:
        return []
    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception:
        return []
    results = []
    for i in range(len(matches) - 1):
        start = float(matches[i][0]) * 1000
        end = float(matches[i + 1][0]) * 1000
        raw = matches[i][1].strip()
        if not raw or not re.search(r'[^ .\n]', raw):
            continue
        cleaned, excl = clean_text(raw)
        cu = cleaned.strip().upper()
        if excl or not cu or cu in normalized_excluded or cu in strict_exclusions:
            continue
        uid = generate_uid()
        try:
            audio[start:end].export(
                os.path.join(AUDIO_OUTPUT_DIR, f'{uid}.wav'), format='wav'
            )
            with open(
                os.path.join(TEXT_OUTPUT_DIR, f'{uid}.txt'),
                'w',
                encoding='utf-8',
            ) as f:
                f.write(cleaned)
            results.append(cleaned)
        except Exception:
            continue
    return results


def main():
    trs_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.trs')]
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_file, f): f for f in trs_files}
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Processing Dataset',
        ):
            pass
    print('UWB dataset processing completed.')


if __name__ == '__main__':
    main()
