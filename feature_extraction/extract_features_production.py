
"""
Production-quality feature extraction script.

Preserves the original feature set:
- duration
- energy
- mean pitch (piptrack)
- MFCC 1-13 means

Outputs: features.csv
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SAMPLE_RATE = 16000
N_MFCC = 13
OUTPUT_CSV = "features.csv"
LOG_FILE = "feature_extraction_errors.log"

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset" / "Audio"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

AUDIO_EXTENSIONS = {".wav", ".mp3"}


def collect_audio_files(dataset_path: Path):
    files = []
    for label_dir in sorted(dataset_path.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name

        for task_dir in sorted(label_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task = task_dir.name

            for audio in sorted(task_dir.iterdir()):
                if audio.suffix.lower() in AUDIO_EXTENSIONS:
                    files.append((audio, label, task))

    return files


def extract_features(item):
    filepath, label, task = item

    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)

        duration = librosa.get_duration(y=y, sr=sr)
        energy = float(np.mean(y ** 2))

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC
        )
        mfcc_mean = np.mean(mfcc, axis=1)

        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr
        )

        pitch_values = pitches[pitches > 0]
        mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) else 0.0

        row = {
            "filename": filepath.name,
            "task": task,
            "label": label,
            "duration": duration,
            "energy": energy,
            "pitch": mean_pitch,
        }

        for i in range(N_MFCC):
            row[f"mfcc_{i+1}"] = float(mfcc_mean[i])

        return row

    except Exception as e:
        logging.exception("Failed processing %s", filepath)
        return None


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print(f"Dataset: {DATASET_PATH}")

    start = time.perf_counter()

    files = collect_audio_files(DATASET_PATH)

    print(f"Found {len(files)} audio files")

    features = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        results = executor.map(extract_features, files)

        for row in tqdm(results, total=len(files), desc="Extracting Features"):
            if row is not None:
                features.append(row)

    features.sort(
        key=lambda x: (
            x["label"],
            x["task"],
            x["filename"]
        )
    )

    df = pd.DataFrame(features)

    columns = (
        ["filename", "task", "label", "duration", "energy", "pitch"]
        + [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
    )

    df = df[columns]

    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.perf_counter() - start

    print("\nFeature Extraction Complete")
    print(f"Files processed : {len(df)}")
    print(f"Errors          : {len(files)-len(df)}")
    print(f"CPU workers     : {os.cpu_count()}")
    print(f"Time elapsed    : {elapsed:.2f} seconds")
    print(f"Output          : {OUTPUT_CSV}")
    print(f"Error log       : {LOG_FILE}")


if __name__ == "__main__":
    main()

