import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE
from config_finetune import (
    TRAIN_CONFIG,
    USE_CONTRASTIVE_LOSS,
    MAX_SEQ_LENGTH
)

from dataset import load_finetune_data, create_dataloaders
from model import FineTunedModel
from trainer import FineTuneTrainer


def main():
    """
    اجرای کامل فرآیند فاین‌تیون
    """
    print("=" * 60)
    print("🚀 Starrt fine_tuning with LoRA (and contrastive learning)")
    print("=" * 60)
    
    # ============================================================
    # Load data
    # ============================================================
    print("\n📊  Load data")
    train_df, test_df = load_finetune_data()
    
    # ============================================================
    # LoRA
    # ============================================================
    print("\n🧠 Create model by LoRA")
    model = FineTunedModel(
        use_contrastive=USE_CONTRASTIVE_LOSS,
        max_length=MAX_SEQ_LENGTH
    )

    # ============================================================
    # Refrence embedding - Semi-Hard Negative
    # ============================================================
    print("\n🔹 Refrence embedding - Semi-Hard Negative")

    model.model.eval()

    reference_embeddings = model.encode(
        train_df["content"].astype(str).tolist(),
        batch_size=16,
        show_progress=True
    ).numpy()

    print(
        f"✅ Reference embeddings shape: "
        f"{reference_embeddings.shape}"
    )
    
    # ============================================================
    # Create DataLoader
    # ============================================================
    print("\n📦  Create DataLoader")
    train_loader, train_dataset = create_dataloaders(
        train_df=train_df,
        tokenizer=model.tokenizer,
        batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        max_length=MAX_SEQ_LENGTH,
        use_contrastive=USE_CONTRASTIVE_LOSS,
        reference_embeddings=reference_embeddings
    )
    
    # ============================================================
    # Create Trainer
    # ============================================================
    print("\n🎯 Create Trainer")
    trainer = FineTuneTrainer(
        model=model,
        train_loader=train_loader,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        num_epochs=TRAIN_CONFIG["num_train_epochs"]
    )
    
    # ============================================================
    # Traning
    # ============================================================
    best_loss = trainer.train()
    
    print("\n" + "=" * 60)
    print("✅ fine_tuning is done successfully")
    print(f"🏆 better loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()