import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json

from config_finetune import (
    FINETUNE_CHECKPOINTS_DIR,
    FINETUNE_LOGS_DIR,
    TRAIN_CONFIG
)


class FineTuneTrainer:
    """
    مربی برای فاین‌تیون مدل با LoRA
    
    این کلاس مسئول:
    - اجرای حلقه آموزش
    - به‌روزرسانی وزن‌ها با Optimizer
    - تنظیم نرخ یادگیری با Scheduler
    - ذخیره checkpointها
    - ثبت لاگ‌ها
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        learning_rate: float = TRAIN_CONFIG["learning_rate"],
        num_epochs: int = TRAIN_CONFIG["num_train_epochs"],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.gradient_accumulation_steps = TRAIN_CONFIG[
            "gradient_accumulation_steps"
        ]
        
        # ============================================================
        # 1. انتقال مدل به دستگاه (GPU/CPU)
        # ============================================================
        self.model.to(device)
        print(f"✅ مدل به {device} منتقل شد.")
        
        # ============================================================
        # 2. تنظیم Optimizer (AdamW)
        # ============================================================
        # AdamW: نسخه بهبودیافته Adam با Weight Decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        
        # ============================================================
        # 3. تنظیم Scheduler (نرخ یادگیری پویا)
        # ============================================================
        # کاهش نرخ یادگیری به صورت خطی بعد از Warmup
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
            num_training_steps=total_steps
        )
        
        # ============================================================
        # 4. تاریخچه برای نمایش و تحلیل
        # ============================================================
        self.loss_history = []
    
    def train_epoch(self, epoch: int) -> float:
        """
        آموزش یک دوره کامل
        
        Args:
            epoch: شماره دوره (برای نمایش)
            
        Returns:
            میانگین در این دوره
        """
        self.model.train()  # حالت آموزش (فعال کردن Dropout)
        total_loss = 0
        
        # Progress Bar برای نمایش پیشرفت
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}"
        )


        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # ============================================================
            # انتقال داده به دستگاه
            # ============================================================
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # ============================================================
            # پیش‌روی مدل (Forward Pass)
            # ============================================================
            if self.model.use_contrastive:
                # حالت با کنتراستی

                positive_input_ids = batch[
                    "positive_input_ids"
                ].to(self.device)

                positive_attention_mask = batch[
                    "positive_attention_mask"
                ].to(self.device)

                negative_input_ids = batch[
                    "negative_input_ids"
                ].to(self.device)

                negative_attention_mask = batch[
                    "negative_attention_mask"
                ].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    positive_input_ids=positive_input_ids,
                    positive_attention_mask=positive_attention_mask,
                    negative_input_ids=negative_input_ids,
                    negative_attention_mask=negative_attention_mask
                )
            else:
                # حالت بدون کنتراستی
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            loss = outputs["loss"]
            
            # ============================================================
            # Gradient accumulation (Backward Pass)
            # ============================================================
            loss = loss / self.gradient_accumulation_steps
            loss.backward()


            
            # ============================================================
            # (Gradient Clipping)
            # ============================================================
            # جلوگیری از انفجار گرادیان
            if (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(progress_bar)
            ):

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=TRAIN_CONFIG["max_grad_norm"]
                )

            
            # ============================================================
            # Update weights
            # ============================================================
            self.optimizer.step()      # Optimizer
            self.scheduler.step()      # learning rate update
            self.optimizer.zero_grad()
            
            # ============================================================
            # save
            # ============================================================
            total_loss += loss.item() * self.gradient_accumulation_steps

            self.loss_history.append(
                loss.item() * self.gradient_accumulation_steps
    )
            
            # ============================================================
            # 8. Update Progress Bar
            # ============================================================
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "avg_loss": f"{total_loss/(batch_idx+1):.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def train(self):
        """
        اجرای کامل فرآیند آموزش
        """
        print(f"\n🚀 شروع آموزش بر روی {self.device}")
        print(f"📊 تعداد نمونه‌های آموزش: {len(self.train_loader.dataset)}")
        print(f"📊 تعداد بچ‌ها: {len(self.train_loader)}")
        print(f"📊 تعداد دوره‌ها: {self.num_epochs}")
        print(f"📊 نرخ یادگیری: {self.learning_rate}")
        print("=" * 60)
        
        best_loss = float('inf')
        
        # ============================================================
        # حلقه آموزش برای هر دوره
        # ============================================================
        for epoch in range(self.num_epochs):
            print(f"\n📚 Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 60)
            
            # آموزش یک دوره
            avg_loss = self.train_epoch(epoch)
            
            # نمایش نتایج
            print(f"\n✅ پایان Epoch {epoch+1}:")
            print(f"   - میانگین: {avg_loss:.4f}")
            print(f"   - نرخ یادگیری فعلی: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # ============================================================
            # ذخیره بهترین مدل (بر اساس کمترین)
            # ============================================================
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = FINETUNE_CHECKPOINTS_DIR / "best_model"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                self.model.model.save_pretrained(checkpoint_path)
                self.model.tokenizer.save_pretrained(checkpoint_path)
                print(f"   - 💾 مدل بهترین ذخیره شد: {checkpoint_path}")
            
            # ============================================================
            # ذخیره Checkpoint دوره
            # ============================================================
            checkpoint_path = FINETUNE_CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch+1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            self.model.model.save_pretrained(checkpoint_path)
            self.model.tokenizer.save_pretrained(checkpoint_path)
            
            # ============================================================
            # ذخیره تاریخچه
            # ============================================================
            FINETUNE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(FINETUNE_LOGS_DIR / "loss_history.json", 'w') as f:
                json.dump(self.loss_history, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ آموزش کامل شد!")
        print(f"🏆 بهترین: {best_loss:.4f}")
        print(f"💾 مدل نهایی در {FINETUNE_CHECKPOINTS_DIR} ذخیره شد.")
        
        return best_loss