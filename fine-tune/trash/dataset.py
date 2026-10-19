import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import random

from config_finetune import (
    FINETUNE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    TEXT_COLUMN,
    CATEGORY_COLUMN,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    NEGATIVE_STRATEGY,
    USE_CONTRASTIVE_LOSS,
    SEMI_HARD_CANDIDATES,
    MIN_NEGATIVE_SIMILARITY
)


class NewsDataset(Dataset):
    """
    دیتاست برای فاین‌تیون با رویکرد کنتراستی
    
    هر نمونه شامل سه بخش است:
    - anchor: متن اصلی (نمونه مرجع)
    - positive: متن مشابه (از همان دسته)
    - negative: متن نامشابه (از دسته دیگر)
    
    این ساختار به مدل یاد می‌دهد که متون مشابه را به هم نزدیک
    و متون نامشابه را از هم دور کند.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_SEQ_LENGTH,
        use_contrastive: bool = USE_CONTRASTIVE_LOSS,
        negative_strategy: str = NEGATIVE_STRATEGY,
        text_column: str = TEXT_COLUMN,
        category_column: str = CATEGORY_COLUMN,
        subcategory_column: str = None,
        reference_embeddings=None,
    ):
        # ذخیره داده‌ها
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_contrastive = use_contrastive
        self.negative_strategy = negative_strategy
        self.text_column = text_column
        self.category_column = category_column
        self.subcategory_column = subcategory_column
        
        self.reference_embeddings = reference_embeddings

        if self.use_contrastive:
            self._build_category_maps()

    
    def _build_category_maps(self):
        """
        ساخت نگاشت دسته‌ها به اندیس نمونه‌ها
        """

        self.category_to_indices = {}
        self.negative_map = {}

        # اندیس‌های هر دسته
        for idx, row in self.df.iterrows():
            category = row[self.category_column]

            if category not in self.category_to_indices:
                self.category_to_indices[category] = []

            self.category_to_indices[category].append(idx)

        # اندیس‌های دسته‌های دیگر برای هر دسته
        all_indices = set(range(len(self.df)))

        for category, indices in self.category_to_indices.items():
            other_indices = list(
                all_indices - set(indices)
            )

            self.negative_map[category] = other_indices

    
    def _get_semi_hard_negative(self, idx: int) -> Dict:
        """
        ایجاد نگاشت از هر دسته به نمونه‌های دسته‌های دیگر
        این کار برای انتخاب سریع نمونه‌های منفی انجام می‌شود
        """
        # برای هر دسته، لیست ایندکس‌های آن دسته را ذخیره می‌کنیم
        row = self.df.iloc[idx]
        category = row[self.category_column]

        other_indices = self.negative_map[category]

        if not other_indices:
            random_idx = random.randint(0, len(self.df) - 1)
            random_row = self.df.iloc[random_idx]

            return {
                "text": random_row[self.text_column],
                "category": random_row[self.category_column]
            }

        # --------------------------------------------------
        # Easy negative
        # --------------------------------------------------

        if self.negative_strategy == "easy":

            negative_idx = random.choice(other_indices)

        # --------------------------------------------------
        # Semi-hard negative
        # --------------------------------------------------

        elif (
            self.negative_strategy == "semi_hard"
            and self.reference_embeddings is not None
        ):

            candidate_count = min(
                SEMI_HARD_CANDIDATES,
                len(other_indices)
            )

            candidate_indices = random.sample(
                other_indices,
                candidate_count
            )

            anchor_embedding = self.reference_embeddings[idx]

            candidate_embeddings = self.reference_embeddings[
                candidate_indices
            ]

            # cosine similarity
            anchor_norm = np.linalg.norm(anchor_embedding)

            candidate_norms = np.linalg.norm(
                candidate_embeddings,
                axis=1
            )

            similarities = (
                candidate_embeddings @ anchor_embedding
            ) / (
                candidate_norms * anchor_norm + 1e-8
            )

            # candidateهایی که similarity قابل قبول دارند
            valid_candidates = [
                (candidate_indices[i], similarities[i])
                for i in range(len(candidate_indices))
                if similarities[i] >= MIN_NEGATIVE_SIMILARITY
            ]

            if valid_candidates:

                # نزدیک‌ترین negative را انتخاب می‌کنیم
                negative_idx = max(
                    valid_candidates,
                    key=lambda x: x[1]
                )[0]

            else:

                # اگر candidate مناسب پیدا نشد،
                # نزدیک‌ترین candidate موجود را بگیر
                negative_idx = candidate_indices[
                    int(np.argmax(similarities))
                ]

        else:

            negative_idx = random.choice(other_indices)

        negative_row = self.df.iloc[negative_idx]

        return {
            "text": negative_row[self.text_column],
            "category": negative_row[self.category_column]
        }    
    def __len__(self):
        """تعداد کل نمونه‌ها"""
        return len(self.df)
    
    def _get_positive_sample(self, idx: int) -> Dict:
        """
        انتخاب یک نمونه مثبت از همان دسته
        
        Args:
            idx: ایندکس نمونه فعلی
            
        Returns:
            دیکشنری شامل متن و دسته نمونه مثبت
        """
        row = self.df.iloc[idx]
        category = row[self.category_column]
        
        # همه نمونه‌های هم‌دسته
        same_category_indices = self.category_to_indices[category]
        
        # حذف خود نمونه (تا خودش را انتخاب نکند)
        same_category_indices = [i for i in same_category_indices if i != idx]
        
        # اگر نمونه دیگری در دسته نیست، خودش را برگردان
        if not same_category_indices:
            return {
                "text": row[self.text_column],
                "category": category
            }
        
        # انتخاب تصادفی از نمونه‌های هم‌دسته
        positive_idx = random.choice(same_category_indices)
        positive_row = self.df.iloc[positive_idx]
        
        return {
            "text": positive_row[self.text_column],
            "category": positive_row[self.category_column]
        }
    
    def _get_negative_sample(self, idx: int) -> Dict:
        """
        انتخاب یک نمونه منفی از دسته دیگر
        
        Args:
            idx: ایندکس نمونه فعلی
            
        Returns:
            دیکشنری شامل متن و دسته نمونه منفی
        """
        row = self.df.iloc[idx]
        category = row[self.category_column]
        
        # همه نمونه‌های دسته‌های دیگر
        other_indices = self.negative_map[category]
        
        # اگر نمونه دیگری در دسته‌های دیگر نیست، یک نمونه تصادفی برگردان
        if not other_indices:
            random_idx = random.randint(0, len(self.df) - 1)
            random_row = self.df.iloc[random_idx]
            return {
                "text": random_row[self.text_column],
                "category": random_row[self.category_column]
            }
        
        # انتخاب تصادفی از نمونه‌های دسته‌های دیگر
        negative_idx = random.choice(other_indices)
        negative_row = self.df.iloc[negative_idx]
        
        return {
            "text": negative_row[self.text_column],
            "category": negative_row[self.category_column]
        }
    
    def __getitem__(self, idx: int) -> Dict:
        """
        بازگرداندن یک نمونه با فرمت مناسب برای مدل
        
        Args:
            idx: ایندکس نمونه
            
        Returns:
            دیکشنری شامل input_ids و attention_mask برای
            anchor, positive, negative
        """
        row = self.df.iloc[idx]
        anchor_text = row[self.text_column]
        
        if self.use_contrastive:
            # انتخاب نمونه مثبت و منفی
            positive = self._get_positive_sample(idx)
            negative = self._get_semi_hard_negative(idx)
            
            # توکنایز کردن هر سه متن
            anchor_enc = self.tokenizer(
                anchor_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            positive_enc = self.tokenizer(
                positive["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            negative_enc = self.tokenizer(
                negative["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": anchor_enc["input_ids"].squeeze(0),
                "attention_mask": anchor_enc["attention_mask"].squeeze(0),
                "positive_input_ids": positive_enc["input_ids"].squeeze(0),
                "positive_attention_mask": positive_enc["attention_mask"].squeeze(0),
                "negative_input_ids": negative_enc["input_ids"].squeeze(0),
                "negative_attention_mask": negative_enc["attention_mask"].squeeze(0),
                "anchor_category": row[self.category_column],
                "positive_category": positive["category"],
                "negative_category": negative["category"]
            }
        else:
            # حالت بدون کنتراستی (فقط متن اصلی)
            enc = self.tokenizer(
                anchor_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "category": row[self.category_column]
            }


def load_finetune_data():
    """
    بارگذاری داده‌های Train و Test برای فاین‌تیون
    
    Returns:
        train_df, test_df: دیتافریم‌های Train و Test
    """
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    print(f"✅ Train: {len(train_df)} سند")
    print(f"✅ Test: {len(test_df)} سند")
    
    print("\n📈 توزیع دسته‌ها در Train:")
    print(train_df[CATEGORY_COLUMN].value_counts())
    
    return train_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = MAX_SEQ_LENGTH,
    use_contrastive: bool = USE_CONTRASTIVE_LOSS,
    reference_embeddings=None
) -> Tuple[DataLoader, Dataset]:
    """
    ایجاد DataLoader برای آموزش
    
    Args:
        train_df: دیتافریم داده‌های Train
        tokenizer: توکنایزر مدل
        batch_size: اندازه بچ
        max_length: حداکثر طول توکن
        use_contrastive: استفاده از کنتراستی
        
    Returns:
        train_loader: DataLoader برای آموزش
        train_dataset: دیتاست آموزشی
    """
    # ایجاد دیتاست
    train_dataset = NewsDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        use_contrastive=use_contrastive,
        reference_embeddings=reference_embeddings
    )
    
    # ایجاد DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # داده‌ها را تصادفی می‌کند
        num_workers=4,         # تعداد پردازش‌های موازی
        pin_memory=True        # انتقال سریع‌تر داده به GPU
    )
    
    return train_loader, train_dataset