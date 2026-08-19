"""
Builds a SentenceTransformer from a raw HuggingFace checkpoint (ParsBERT
is not natively a sentence-transformers model) and fine-tunes it with
supervised batch-hard triplet loss using category labels.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments,
    BatchSamplers,
)
from sentence_transformers.trainer import SentenceTransformerTrainer

from config import (
    FINETUNE_MODEL_NAME,
    FINETUNED_MODEL_DIR,
    MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

# Effective batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS.
# On a 4GB GPU (RTX 3050), TRAIN_BATCH_SIZE=16 with accumulation=2 keeps
# the same effective batch of 32 used for the loss/gradient computation,
# while only ever holding 16 examples in memory at once.
GRADIENT_ACCUMULATION_STEPS = 2


def build_sentence_transformer(model_name=FINETUNE_MODEL_NAME,
                                max_seq_length=MAX_SEQ_LENGTH):
    """
    ParsBERT is a raw HF encoder with no pooling layer defined for sentence
    embeddings. Build one manually: token embeddings -> mean pooling ->
    single sentence vector. Mean pooling is the standard choice here.
    """

    word_embedding_model = models.Transformer(
        model_name,
        max_seq_length=max_seq_length,
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def build_training_args(output_dir=str(FINETUNED_MODEL_DIR),
                         batch_size=TRAIN_BATCH_SIZE,
                         epochs=EPOCHS,
                         learning_rate=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY,
                         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS):
    """
    batch_sampler=GROUP_BY_LABEL replaces the old SentenceLabelDataset --
    it guarantees each batch contains multiple docs per category, which
    BatchHardTripletLoss needs in order to mine same-category pairs.

    fp16=True halves activation memory on Ampere GPUs (RTX 30-series)
    with negligible accuracy cost -- needed here given the 4GB VRAM budget.

    load_best_model_at_end / metric_for_best_model deliberately omitted:
    with only a custom evaluator (no eval_dataset), transformers' best-
    checkpoint tracking is unreliable across versions. The evaluator's
    printed KNN accuracy at each eval_steps interval is the real signal
    to watch; the final saved model is simply the last-epoch checkpoint.
    """

    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

def train(model, train_dataset, evaluator=None, output_dir=str(FINETUNED_MODEL_DIR)):

    train_loss = losses.BatchHardTripletLoss(model=model)
    args = build_training_args(output_dir=output_dir)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()

    # Explicit save -- belt and braces on top of load_best_model_at_end,
    # so the final checkpoint on disk is guaranteed to be the best one.
    model.save(output_dir)
    print(f"Fine-tuned model saved to: {output_dir}")

    return model