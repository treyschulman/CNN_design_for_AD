import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import shutil

# ==== CONFIG ====
csv_path = "oasis_cross-sectional-5708aa0a98d82080.csv"
image_root = "oasis"  # directory containing disc1/, disc2/, ..., disc10/
output_root = "dataset_split"  # where to save train/val/test folders
os.makedirs(output_root, exist_ok=True)

# ==== STEP 1: Load clinical data and map CDR to diagnosis ====
df = pd.read_csv(csv_path)
df = df[~df['CDR'].isna()]

def map_diagnosis(cdr):
    if cdr == 0.0:
        return 'CN'
    elif cdr == 0.5:
        return 'MCI'
    else:
        return 'AD'

df['diagnosis'] = df['CDR'].apply(map_diagnosis)
df_unique = df.drop_duplicates(subset='ID')

# ==== STEP 2: Stratified split (by subject ID) ====
train_val_ids, test_ids = train_test_split(
    df_unique, test_size=0.15, stratify=df_unique['diagnosis'], random_state=42
)
train_ids, val_ids = train_test_split(
    train_val_ids, test_size=0.1765, stratify=train_val_ids['diagnosis'], random_state=42
)
# 70/15/15 ratio

id_to_split = {
    id_: 'Train' for id_ in train_ids['ID']
}
id_to_split.update({id_: 'Val' for id_ in val_ids['ID']})
id_to_split.update({id_: 'Test' for id_ in test_ids['ID']})

# ==== STEP 3: Copy subjects into split folders ====
for disc in sorted(os.listdir(image_root)):
    disc_path = os.path.join(image_root, disc)
    if not os.path.isdir(disc_path):
        continue
    for subj_dir in os.listdir(disc_path):
        if not subj_dir.startswith("OAS1_") or not subj_dir.endswith("_MR1"):
            continue
        try:
            subj_id_str = subj_dir.split("_")[1]
            subj_id = int(subj_id_str)
        except Exception:
            continue

        if subj_id not in id_to_split:
            continue

        split = id_to_split[subj_id]
        src_path = os.path.join(disc_path, subj_dir)
        dst_path = os.path.join(output_root, split.lower(), subj_dir)

        if not os.path.exists(dst_path):
            shutil.copytree(src_path, dst_path)
            print(f"Copied: {subj_dir} -> {split}")

print("✅ Finished. Check:", output_root)
