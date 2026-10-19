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

import gc
import torch
import numpy as np
from config import CATEGORY_COLUMN, FINETUNED_MODEL_DIR, SEEDS
from training.dataset_builder import load_finetune_data
from training.embedding_trainer import build_sentence_transformer, train
from training.evaluator import KNNCategoryEvaluator



def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_seed(seed, label_column=CATEGORY_COLUMN):
    """یک اجرای کامل fine-tuning با seed مشخص."""

    set_seed(seed)

    output_dir = FINETUNED_MODEL_DIR / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_texts, val_labels, label_encoder = load_finetune_data(
        label_column=label_column
    )

    model = build_sentence_transformer()
    evaluator = KNNCategoryEvaluator(val_texts, val_labels)

    train(model, train_dataset, evaluator=evaluator, output_dir=str(output_dir))

    # پاکسازی حافظه بین seedها
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return str(output_dir)


def main():
    results = {}
    for seed in SEEDS:
        print(f"\n{'='*60}\nRunning seed {seed}\n{'='*60}")
        model_dir = run_one_seed(seed)
        results[seed] = model_dir

    return results


if __name__ == "__main__":
    main()
