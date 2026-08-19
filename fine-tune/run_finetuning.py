"""
Entry point for embedding fine-tuning. Call main() from 02_finetuning.ipynb.

Only train.csv is used here. test.csv is never touched -- it's reserved
for 03_comparison.ipynb so the final NMI/purity/coherence numbers are a
fair, leakage-free comparison against baseline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from config import CATEGORY_COLUMN, FINETUNED_MODEL_DIR
from training.dataset_builder import load_finetune_data
from training.embedding_trainer import build_sentence_transformer, train
from training.evaluator import KNNCategoryEvaluator


def main(label_column=CATEGORY_COLUMN):

    FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_texts, val_labels, label_encoder = load_finetune_data(
        label_column=label_column
    )

    model = build_sentence_transformer()
    evaluator = KNNCategoryEvaluator(val_texts, val_labels)

    train(model, train_dataset, evaluator=evaluator, output_dir=str(FINETUNED_MODEL_DIR))

    return model, label_encoder


if __name__ == "__main__":
    main()