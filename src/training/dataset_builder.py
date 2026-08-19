"""
Builds labeled training/validation examples for embedding fine-tuning.

Only the TRAIN split (train.csv) is touched here. test.csv stays untouched
until 03_comparison.ipynb, so the final evaluation metrics are never
computed on data the fine-tuned embedding model has already seen.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import TRAIN_DATA_PATH, TEXT_COLUMN, CATEGORY_COLUMN, RANDOM_STATE

# BatchHardTripletLoss needs >=2 examples of the same label inside a batch
# to mine a positive pair. Categories with too few docs get dropped rather
# than silently breaking batches.
MIN_DOCS_PER_LABEL = 20


def _filter_rare_labels(df, label_column, min_count=MIN_DOCS_PER_LABEL):
    counts = df[label_column].value_counts()
    valid_labels = counts[counts >= min_count].index
    dropped = counts[counts < min_count]
    if len(dropped) > 0:
        print(f"Dropping {len(dropped)} rare '{label_column}' labels "
              f"(< {min_count} docs): {list(dropped.index)}")
    return df[df[label_column].isin(valid_labels)].reset_index(drop=True)


def load_finetune_data(label_column=CATEGORY_COLUMN, val_size=0.1):
    """
    Loads train.csv, filters rare labels, splits into a fine-tuning train
    set and a small internal validation set (stratified), and encodes
    labels to integers.

    Returns
    -------
    train_dataset  : datasets.Dataset with columns "sentence", "label"
    val_texts      : list[str]
    val_labels     : list[int]
    label_encoder  : sklearn.preprocessing.LabelEncoder
    """

    df = pd.read_csv(TRAIN_DATA_PATH)
    df = df.dropna(subset=[TEXT_COLUMN, label_column])
    df = _filter_rare_labels(df, label_column)

    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df[label_column])

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )

    train_dataset = Dataset.from_dict({
        "sentence": train_df[TEXT_COLUMN].astype(str).tolist(),
        "label": train_df["label_id"].tolist(),
    })

    val_texts = val_df[TEXT_COLUMN].astype(str).tolist()
    val_labels = val_df["label_id"].tolist()

    print(f"Fine-tune train examples: {len(train_dataset)}")
    print(f"Fine-tune val examples:   {len(val_texts)}")
    print(f"Number of classes:        {len(encoder.classes_)}")

    return train_dataset, val_texts, val_labels, encoder