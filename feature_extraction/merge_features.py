import os
import pandas as pd

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ACOUSTIC_CSV = os.path.join(BASE_DIR, "features.csv")
LINGUISTIC_CSV = os.path.join(BASE_DIR, "linguistic_features.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "merged_features.csv")

# =====================================================
# CHECK FILES
# =====================================================

if not os.path.exists(ACOUSTIC_CSV):
    raise FileNotFoundError(f"Could not find {ACOUSTIC_CSV}")

if not os.path.exists(LINGUISTIC_CSV):
    raise FileNotFoundError(f"Could not find {LINGUISTIC_CSV}")

# =====================================================
# LOAD DATA
# =====================================================

print("Loading datasets...")

acoustic_df = pd.read_csv(ACOUSTIC_CSV)
linguistic_df = pd.read_csv(LINGUISTIC_CSV)

print(f"Acoustic Samples   : {len(acoustic_df)}")
print(f"Linguistic Samples : {len(linguistic_df)}")

# =====================================================
# MAKE FILENAMES IDENTICAL
# =====================================================

# Remove file extensions
acoustic_df["filename"] = (
    acoustic_df["filename"]
    .astype(str)
    .str.replace(".mp3", "", regex=False)
    .str.replace(".wav", "", regex=False)
)

linguistic_df["filename"] = (
    linguistic_df["filename"]
    .astype(str)
    .str.replace(".mp3", "", regex=False)
    .str.replace(".wav", "", regex=False)
)

# Remove leading/trailing spaces
acoustic_df["filename"] = acoustic_df["filename"].str.strip()
linguistic_df["filename"] = linguistic_df["filename"].str.strip()

# =====================================================
# MERGE
# =====================================================

merged_df = acoustic_df.merge(
    linguistic_df,
    on=["filename", "label", "task"],
    how="inner"
)

# =====================================================
# SAVE
# =====================================================

merged_df.to_csv(OUTPUT_CSV, index=False)

print("\nMerge Complete!")

print(f"Merged Samples : {len(merged_df)}")
print(f"Saved to : {OUTPUT_CSV}")

# =====================================================
# CHECK UNMATCHED FILES
# =====================================================

missing_audio = set(acoustic_df["filename"]) - set(merged_df["filename"])
missing_text = set(linguistic_df["filename"]) - set(merged_df["filename"])

print("\n==============================")

if missing_audio:
    print(f"Audio files without transcript features : {len(missing_audio)}")
else:
    print("All acoustic files matched successfully.")

if missing_text:
    print(f"Transcript files without acoustic features : {len(missing_text)}")
else:
    print("All transcript files matched successfully.")

print("==============================")

print("\nMerged Data Preview:")
print(merged_df.head())