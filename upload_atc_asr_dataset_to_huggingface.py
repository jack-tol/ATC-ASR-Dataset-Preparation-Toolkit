import os
from datasets import Dataset, DatasetDict, Audio

def load_split_data(split_folder):
    transcripts_folder = os.path.join(split_folder, "texts")
    audios_folder = os.path.join(split_folder, "audios")
    ids, audio_paths, texts = [], [], []
    for transcript_file in os.listdir(transcripts_folder):
        if transcript_file.endswith(".txt"):
            uid = os.path.splitext(transcript_file)[0]
            transcript_path = os.path.join(transcripts_folder, transcript_file)
            audio_path = os.path.join(audios_folder, f"{uid}.wav")
            if os.path.exists(audio_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                ids.append(uid)
                audio_paths.append(audio_path)
                texts.append(text)
    return Dataset.from_dict({"id": ids, "audio": audio_paths, "text": texts}).cast_column("audio", Audio())

base_path = "ATC_ASR_Dataset_Splits"

train_dataset = load_split_data(os.path.join(base_path, "train"))
validation_dataset = load_split_data(os.path.join(base_path, "validation"))
test_dataset = load_split_data(os.path.join(base_path, "test"))

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

dataset_dict.push_to_hub("ATC_ASR_Dataset", private=True)