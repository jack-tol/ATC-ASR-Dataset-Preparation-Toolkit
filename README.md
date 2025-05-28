# ATC ASR Dataset Preparation Toolkit

This repository contains scripts for transforming three distinct raw Air Traffic Control (ATC) text-speech corpora into a single, high-quality, fine-tuning-ready dataset for Automatic Speech Recognition (ASR).

Each script is responsible for segmenting long recordings, filtering out unusable utterances, normalizing the text, and generating aligned `wav + txt` pairs at the utterance level.

In addition, the repository includes tools to further process the unified dataset, such as splitting, augmenting, and uploading it to the Hugging Face Hub, making it readily accessible for ASR training and fine-tuning.

You can find the **ATC ASR Dataset**, which includes cleaned and combined data from the UWB and ATCO2 corpora, uploaded to Hugging Face Datasets here: [ATC_ASR_Dataset on Hugging Face](https://huggingface.co/datasets/jacktol/ATC_ASR_Dataset).

You can find download links and further details on each source corpus in the [Supported Datasets](#supported-datasets) section.

## The Problem

Out of the box, the raw datasets present two major challenges.

First, none of the datasets are properly segmented into transmission–text pairs, which is essential for effective ASR training or fine-tuning. Instead, they typically consist of multiple XML or TXT files, each containing several transcripts annotated with start and stop timestamps, paired with corresponding long audio recordings, ranging from several minutes to multiple hours in duration. These unsegmented files make it difficult to extract clean, per-utterance examples suitable for training.

Second, and particularly problematic in the case of the UWB ATC dataset, the transcripts are extremely unclean. Common issues include:

- Numerous spelling mistakes
- Corrupted or non-Unicode characters
- Symbols and broken formatting
- Inconsistent capitalization
- Missing phonetic expansions (e.g., "N" instead of "NOVEMBER")
- Numeric digits instead of their word equivalents (e.g., "350" instead of "THREE FIVE ZERO")
- Nested parentheses indicating intended vs. pronounced words
- Inconsistent tagging schemes scattered throughout
- Transmissions mixing English with non-English words in the same sentence

These inconsistencies make the raw data unsuitable for direct use in ASR pipelines without substantial preprocessing and normalization.

### Illustrative Examples | Raw UWB ATC Dataset

Raw UWB transcripts - irrelevant tags, broken characters, mixed language, raw numbers, inconsistent capitalization, unphonetized letters, etc.:

```
[air]Praha radar[czech_|] dobr� den[|_czech] [unintelligible] 9 0 W level t+
[ground]Skyshuttle 1 1 4 0 contact (Vienna(v�n)) 1 3 4  4 4 0 bye bye[speaker]
[ground]good afternoon Lufthansa 7 3 9 Praha radar contact descend FL 3 0 0 level by RAPET
[ground]Lufthansa 3 C N contact Bratislava 1 2 5 . 9 6 5 good bye
```

## Solution

To address the issues present in the raw datasets and produce a clean, ASR-ready dataset, three dedicated Python scripts were developed:

- `dataset_processing_scripts/process_atcc_dataset.py`
- `dataset_processing_scripts/process_atco2_dataset.py`
- `dataset_processing_scripts/process_uwb_dataset.py` 

Each script takes as input the path to the raw dataset as downloaded and performs the necessary preprocessing steps to produce high-quality `wav + txt` utterance pairs.

When required, the scripts apply:

- General spelling corrections
- Diacritic normalization
- Letter-to-phonetic word mappings (e.g., "N" → "NOVEMBER")
- Number-to-word conversions (e.g., "3 5 0" → "THREE FIVE ZERO")
- Capitalization normalization (all text converted to uppercase)
- In-place removal of non-critical tags (e.g., `[GROUND]`, `[AIR]`, `[SPEAKER]`)
- Exclusion of transmissions containing certain tags or conditions (e.g., `[|_NO_ENG]`, `[CZECH_]`, `[|_UNINTELLIGIBLE]`)
- Manual removal of specific transmissions with incorrect or unreliable ground truths

These steps help ensure that the final data is consistent, clean, and ready for downstream ASR model training.

## Supported Datasets

This repository includes processing scripts for the following Air Traffic Control (ATC) speech-text corpora:

### University of West Bohemia (UWB) ATC Corpus

The UWB ATC dataset contains approximately 20 hours of manually transcribed air traffic control communications. The transcriptions were performed by non-native English speakers and reflect a wide range of heavily accented English, as typical of European ATC environments. The recordings come from Czech airspace, and the dataset includes numerous linguistic artifacts such as code-switching, transcription inconsistencies, and phonetic variability, making it a valuable but challenging resource for ASR training.

**Download:** [UWB ATC Corpus](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0)

### Air Traffic Control Complete Corpus (ATCC, LDC94S14A)

The ATCC dataset, distributed by the Linguistic Data Consortium (LDC), comprises digitized recordings of FAA air traffic control radio communications from December 1993. The data was collected at three major U.S. airports: Dallas/Fort Worth International (DFW), Boston Logan International (BOS), and Washington National (DCA). Each audio file spans 1–2 hours and includes both speech and silence. All recordings are manually transcribed and provide a clean, native-English reference set for training and evaluation.

**Purchase/Access:** [ATCC Corpus (LDC94S14A)](https://catalog.ldc.upenn.edu/LDC94S14A)

### ATCO2 Test Subset

This dataset is a publicly available 1-hour test subset released as part of the broader ATCO2 project. While the full ATCO2 corpus contains over 4,465 hours of English ATC speech data, it is not freely available. This smaller test portion, however, is openly accessible and provides a representative sample of the larger corpus. It includes a range of accents, speaker types (pilots and controllers), and environments, and serves as a valuable benchmarking resource for evaluating ASR models on real-world ATC audio.

**Download:** [ATCO2 1-Hour Test Subset](https://www.replaywell.com/atco2/download/ATCO2-ASRdataset-v1_beta.tgz)

## Creating the Combined Dataset

The script `dataset_processing_scripts/create_combined_atc_asr_dataset.py` merges the outputs of the processed datasets (ATCC, ATCO2, and UWB) into a single unified dataset called `ATC_ASR_Dataset`.

This combined dataset contains:

- An `audios/` directory with all audio files
- A `texts/` directory with the corresponding transcripts

Each file pair is matched by a unique ID. During this process, all audio is resampled to 16,000 Hz to ensure consistency across sources and compatibility with standard ASR pipelines.

## Additional Scripts: Splitting, Augmenting, and Uploading

To further prepare the combined dataset for model training, the repository includes additional utility scripts.

### `utils/split_atc_asr_dataset.py`

This script takes the combined `ATC_ASR_Dataset` and performs an 80-10-10 split into training, validation, and test sets. The split is saved under `ATC_ASR_Dataset_Splits`, preserving consistent audio–transcript pairings. Pre-splitting the data ensures that only the training set is augmented in the next stage.

### `utils/offline_data_augmentation.py`

This script augments 50% of the training data in `ATC_ASR_Dataset_Splits/train`. For each selected audio sample, it generates three new augmented versions using between two and three simultaneous augmentation techniques (such as pitch shifting, noise injection, and bandpass filtering). Once complete, the original training set is replaced with its augmented counterpart.

### `utils/upload_dataset_to_huggingface.py`

This script uploads the final dataset, consisting of the train, validation, and test splits, from the `ATC_ASR_Dataset_Splits` directory to the Hugging Face Hub.

Since offline data augmentation is applied to the training set prior to this step, the train split being uploaded includes the augmented data. The script reads all audio–transcript pairs for each split, constructs a `DatasetDict`, and pushes the dataset to the Hugging Face Hub using the `datasets` library.

All audio files are cast as `datasets.Audio` objects, ensuring compatibility with Hugging Face's ASR pipelines. By default, the dataset is uploaded as private. This can be changed by setting `private=False` in the `push_to_hub()` call.

## Related Work & Improvements

This toolkit builds upon prior work by [Juan Pablo Zuluaga](https://github.com/idiap/atco2-corpus/tree/main/data/databases/uwb_atcc), who published a processing script and corresponding Hugging Face dataset for the UWB ATC corpus:
[UWB ATC Dataset on Hugging Face](https://huggingface.co/datasets/Jzuluaga/uwb_atcc)

While that effort helped make the UWB dataset more accessible to the ASR community, this repository adopts a more rigorous approach to data cleaning and preparation.

Key differences in this implementation include:

- Capitalization normalization: All output transcripts are fully capitalized to maintain consistency across datasets.
- Stricter tag handling: Transmissions containing tags such as `[UNINTELLIGIBLE]`, `[|_NO_ENG]`, or `[CZECH_]` are excluded entirely, rather than simply removing the tag. This avoids including transcripts that do not match their corresponding audio.
- Manual filtering: A set of transmissions was manually reviewed and excluded due to confirmed issues with alignment or transcription accuracy.
- Modern, Python-based architecture: This repository is implemented entirely in Python, with modular and readable scripts. Mapping dictionaries, correction lists, and exclusion criteria are centralized in a `dataset_processing_scripts/utils.py` module and applied via clear function calls. This structure improves transparency and maintainability.

The goal of these changes is to prioritize data quality, reproducibility, and developer usability. The result is a cleaner and more reliable dataset for ASR model development, particularly in domain-specific applications like air traffic communication.

## Conclusion

This toolkit provides a full, end-to-end pipeline for transforming noisy, unstructured Air Traffic Control (ATC) speech corpora into a clean, high-quality dataset ready for Automatic Speech Recognition (ASR) training and fine-tuning.

From raw data ingestion and normalization to augmentation and publishing, each step has been designed to ensure consistency, scalability, and compatibility with modern ASR frameworks. The final dataset is aligned, resampled, split, and optionally augmented, resulting in a ready-to-use, Hugging Face-compatible format.

Whether you are building ASR models for aviation-specific applications or experimenting with accented, domain-specific English, this dataset preparation pipeline offers a strong, reproducible foundation.

## References

- [ATC_ASR_Dataset (Combined and Cleaned Dataset)](https://huggingface.co/datasets/jacktol/ATC_ASR_Dataset)
- [ATCC Corpus (LDC94S14A, Raw)](https://catalog.ldc.upenn.edu/LDC94S14A)
- [ATCO2 1-Hour Test Subset (Raw)](https://www.replaywell.com/atco2/download/ATCO2-ASRdataset-v1_beta.tgz)
- [Juan Pablo Zuluaga – UWB ATC Dataset on GitHub](https://github.com/idiap/atco2-corpus/tree/main/data/databases/uwb_atcc)
- [Juan Pablo Zuluaga – UWB ATC Dataset on Hugging Face](https://huggingface.co/datasets/Jzuluaga/uwb_atcc)
- [UWB ATC Corpus (Raw)](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0)