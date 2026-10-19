import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model
)
from typing import Dict, Optional

from config_finetune import (
    FINETUNE_MODEL_NAME,
    LORA_CONFIG,
    USE_CONTRASTIVE_LOSS,
    CONTRASTIVE_TEMPERATURE,
    CONTRASTIVE_LOSS_WEIGHT,
    MAX_SEQ_LENGTH
)


class ContrastiveLoss(nn.Module):
    """
    کنتراستی (InfoNCE)
    
    این به مدل یاد می‌دهد که:
    - نمونه‌های مشابه (anchor و positive) را به هم نزدیک کند
    - نمونه‌های نامشابه (anchor و negative) را از هم دور کند
    
    فرمول: -log(exp(sim(anchor, positive)/τ) / (exp(sim(anchor, positive)/τ) + exp(sim(anchor, negative)/τ)))
    """
    
    def __init__(self, temperature: float = CONTRASTIVE_TEMPERATURE):
        super().__init__()
        self.temperature = temperature  # دمای نرم‌سازی (کنترل تیز بودن توزیع)
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        محاسبه کنتراستی
        
        Args:
            anchor_embeddings: بازنمایی نمونه مرجع
            positive_embeddings: بازنمایی نمونه مثبت (مشابه)
            negative_embeddings: بازنمایی نمونه منفی (نامشابه)
        
        Returns:
            مقدار
        """
        # نرمال‌سازی بردارها (برای محاسبه شباهت کسینوسی)
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
        
        # شباهت با نمونه مثبت (هرچه بیشتر بهتر)
        pos_sim = (anchor_embeddings * positive_embeddings).sum(dim=1) / self.temperature
        
        # شباهت با نمونه منفی (هرچه کمتر بهتر)
        neg_sim = (anchor_embeddings * negative_embeddings).sum(dim=1) / self.temperature
        
        # کنتراستی
        loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))
        )
        
        return loss.mean()


class FineTunedModel(nn.Module):
    """
    مدل فاین‌تیون‌شده با LoRA و امکان یادگیری کنتراستی
    
    این مدل از دو تکنیک ترکیبی استفاده می‌کند:
    1. LoRA: کاهش تعداد پارامترهای قابل آموزش
    2. Contrastive Learning: بهبود کیفیت بازنمایی‌ها
    """
    
    def __init__(
        self,
        model_name: str = FINETUNE_MODEL_NAME,
        use_contrastive: bool = USE_CONTRASTIVE_LOSS,
        max_length: int = MAX_SEQ_LENGTH
    ):
        super().__init__()
        
        self.use_contrastive = use_contrastive
        self.max_length = max_length
        
        # ============================================================
        # 1. بارگذاری مدل پایه ParsBERT
        # ============================================================
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        # ============================================================
        # 2. اعمال LoRA
        # ============================================================
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],                    # رتبه
            lora_alpha=LORA_CONFIG["lora_alpha"],  # ضریب مقیاس
            target_modules=LORA_CONFIG["target_modules"],  # لایه‌های هدف
            lora_dropout=LORA_CONFIG["lora_dropout"],      # Dropout
            bias=LORA_CONFIG["bias"],              # آموزش bias
            task_type=LORA_CONFIG["task_type"]     # نوع وظیفه
        )
        
        # اعمال LoRA روی مدل
        self.model = get_peft_model(self.model, lora_config)
        
        # نمایش تعداد پارامترهای قابل آموزش
        print("\n📊 تعداد پارامترهای مدل:")
        self.model.print_trainable_parameters()
        
        # ============================================================
        # 3. بارگذاری توکنایزر
        # ============================================================
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name , 
            local_files_only=True
            )
        
        # تنظیم توکن padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ============================================================
        # 4. کنتراستی (در صورت استفاده)
        # ============================================================
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss()


    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9
        )

    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positive_input_ids: Optional[torch.Tensor] = None,
        positive_attention_mask: Optional[torch.Tensor] = None,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        پیش‌روی مدل
        
        Args:
            input_ids: توکن‌های نمونه مرجع
            attention_mask: ماسک نمونه مرجع
            positive_input_ids: توکن‌های نمونه مثبت
            positive_attention_mask: ماسک نمونه مثبت
            negative_input_ids: توکن‌های نمونه منفی
            negative_attention_mask: ماسک نمونه منفی
        
        Returns:
            دیکشنری شامل و بازنمایی‌ها
        """
        # ============================================================
        # بازنمایی نمونه مرجع
        # ============================================================
        anchor_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        anchor_embeddings = self.mean_pooling(
            anchor_outputs,
            attention_mask
        )
        
        # ============================================================
        # حالت با کنتراستی
        # ============================================================
        if self.use_contrastive and all([
            positive_input_ids is not None,
            negative_input_ids is not None
        ]):
            # بازنمایی نمونه مثبت
            positive_outputs = self.model(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask
            )
            positive_embeddings = self.mean_pooling(
                positive_outputs,
                positive_attention_mask
            )
            
            # بازنمایی نمونه منفی
            negative_outputs = self.model(
                input_ids=negative_input_ids,
                attention_mask=negative_attention_mask
            )
            negative_embeddings = self.mean_pooling(
                negative_outputs,
                negative_attention_mask
            )
            
            # محاسبه کنتراستی
            contrastive_loss = self.contrastive_loss(
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings
            )
            
            # کل (با وزن)
            loss = contrastive_loss * CONTRASTIVE_LOSS_WEIGHT
            
            return {
                "loss": loss,
                "anchor_embeddings": anchor_embeddings,
                "positive_embeddings": positive_embeddings,
                "negative_embeddings": negative_embeddings,
                "contrastive_loss": contrastive_loss
            }
        
        # ============================================================
        # حالت بدون کنتراستی
        # ============================================================
        return {
            "embeddings": anchor_embeddings,
            "pooler_output": anchor_outputs.pooler_output,
            "last_hidden_state": anchor_outputs.last_hidden_state
        }
    
    def encode(
        self,
        texts: list,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        تولید بازنمایی برای لیستی از متون
        
        این متد بعد از فاین‌تیون برای تولید Embedding استفاده می‌شود
        
        Args:
            texts: لیست متون
            batch_size: اندازه بچ
            show_progress: نمایش پیشرفت
        
        Returns:
            تنسور شامل بازنمایی‌های متون
        """
        self.eval()  # حالت ارزیابی (غیرفعال کردن Dropout و BatchNorm)

        device = next(self.model.parameters()).device
        
        embeddings = []
        
        # پردازش به صورت بچ
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # توکنایز کردن
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # انتقال به GPU (در صورت وجود)
            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }
            
            # تولید بازنمایی
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.pooler_output.cpu()
                embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
