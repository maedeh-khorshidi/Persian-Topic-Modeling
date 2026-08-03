import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)


from trainer import ContrastiveTrainer

from config import (
    DATASET_PATH,
    FINETUNED_MODEL_DIR,
    TRAIN_BATCH_SIZE,
    EPOCHS,
    CATEGORY_COLUMN,
    SUBCATEGORY_COLUMN,
    RANDOM_STATE,
    PROCESSED_DATA_DIR
)

import random
import numpy as np
import torch

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Dataset size: {df.shape}"
    )

    # ---------------------------------
    # Train / Test Split
    # ---------------------------------

    print("Loading train dataset...")

    train_df = pd.read_csv(
        PROCESSED_DATA_DIR / "train.csv"
    )

    print("Loading test dataset...")

    test_df = pd.read_csv(
        PROCESSED_DATA_DIR / "test.csv"
    )

    print(f"Train size: {train_df.shape}")
    print(f"Test size : {test_df.shape}")

    # df["stratify_label"] = (
    # df[CATEGORY_COLUMN] + "__" + df[SUBCATEGORY_COLUMN]
    # )

    # train_df, test_df = train_test_split(

    #     df,

    #     test_size=0.2,

    #     random_state=RANDOM_STATE,

    #     stratify=df["stratify_label"]

    # )

    # train_df = train_df.drop(columns=["stratify_label"])
    # test_df = test_df.drop(columns=["stratify_label"])

    # train_df = train_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)

    # print(
    #     f"Train size: {train_df.shape}"
    # )

    # print(
    #     f"Test size : {test_df.shape}"
    # )

    # train_df.to_csv(
    #     PROCESSED_DATA_DIR/"train.csv",
    #     index=False
    # )

    # test_df.to_csv(
    #     PROCESSED_DATA_DIR/"test.csv",
    #     index=False
    # )


    # ---------------------------------
    # Fine-tuning
    # ---------------------------------

    trainer = ContrastiveTrainer(

        dataframe=train_df,

        batch_size=TRAIN_BATCH_SIZE,

        epochs=EPOCHS,

        output_path=FINETUNED_MODEL_DIR

    )

    trainer.train()

    print("Fine-tuning completed.")


if __name__ == "__main__":

    main()