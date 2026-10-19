"""
Fine-tunes ParsBERT (+ mean pooling) with supervised batch-hard triplet loss.
Best checkpoint is selected by validation k-NN accuracy; early stopping is
applied on the same metric.
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
from transformers import TrainerCallback

from config import (
    FINETUNE_MODEL_NAME, FINETUNED_MODEL_DIR, MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO,
    GRADIENT_ACCUMULATION_STEPS, EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD, METRIC_FOR_BEST_MODEL, GREATER_IS_BETTER,
    LOAD_BEST_MODEL_AT_END,
)


def build_sentence_transformer(model_name=FINETUNE_MODEL_NAME,
                               max_seq_length=MAX_SEQ_LENGTH):
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def build_training_args(output_dir=str(FINETUNED_MODEL_DIR), seed=42,
                        batch_size=TRAIN_BATCH_SIZE, epochs=EPOCHS,
                        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS):
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=WARMUP_RATIO,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        seed=seed,                      # <-- بدون این، هر سه اجرا با seed=42 آموزش می‌بینند
        data_seed=seed,                 # <-- ترتیب/ترکیب batchها
        eval_strategy="epoch",          # <-- هر epoch، نه هر ۱۰۰ گام
        save_strategy="epoch",          # باید با eval_strategy یکی باشد
        save_total_limit=3,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,
        logging_steps=50,
        report_to="none",
    )


class ThresholdEarlyStopping(TrainerCallback):
    """Stop when the metric has not improved by more than `threshold`
    for `patience` consecutive evaluations (higher is better)."""

    def __init__(self, metric_name, patience=2, threshold=0.001):
        self.metric_name = metric_name
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.counter = 0
        self.stopped_at_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return
        current = metrics[self.metric_name]
        if self.best_metric is None or current > self.best_metric + self.threshold:
            self.best_metric = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
                self.stopped_at_epoch = state.epoch
                print(f"🛑 Early stopping at epoch {state.epoch:.0f}: no improvement "
                      f"> {self.threshold} for {self.patience} evaluations")


def train(model, train_dataset, evaluator, output_dir=str(FINETUNED_MODEL_DIR), seed=42):
    train_loss = losses.BatchHardTripletLoss(model=model)
    try:
        loss_cfg = train_loss.get_config_dict()
    except AttributeError:
        loss_cfg = {"margin": getattr(train_loss, "triplet_margin", None),
                    "distance": getattr(train_loss.distance_metric, "__name__", None)}
    print("Loss config:", loss_cfg)          # مقدار واقعی margin و فاصله برای مقاله

    args = build_training_args(output_dir=output_dir, seed=seed)
    early_stop = ThresholdEarlyStopping(
        METRIC_FOR_BEST_MODEL, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset,
        loss=train_loss, evaluator=evaluator, callbacks=[early_stop],
    )
    trainer.train()                 # در پایان، بهترین checkpoint در model بارگذاری می‌شود
    model.save(output_dir)

    history = [
        {"epoch": h["epoch"], "step": h["step"], METRIC_FOR_BEST_MODEL: h[METRIC_FOR_BEST_MODEL]}
        for h in trainer.state.log_history if METRIC_FOR_BEST_MODEL in h
    ]
    info = {
        "seed": seed,
        "best_val_accuracy": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "epochs_trained": trainer.state.epoch,
        "optimizer_steps": trainer.state.global_step,
        "stopped_early": early_stop.stopped_at_epoch is not None,
        "optimizer": str(args.optim),
        "lr_scheduler": str(args.lr_scheduler_type),
        "loss_config": str(loss_cfg),
    }
    return model, info, history