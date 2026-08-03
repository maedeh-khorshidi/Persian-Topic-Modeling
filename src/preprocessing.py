import re
import string
from hazm import (
    Normalizer,
    word_tokenize,
    stopwords_list,
    Lemmatizer
)
import stopwordsiso as stopwords
from config import STOPWORDS_PATH

# Persian normalizer
normalizer = Normalizer()

# Persian stopwords
with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    news_stopwords = {
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    }

base_stopwords  = set(stopwords.stopwords("fa"))

persian_stopwords =news_stopwords | base_stopwords

# Lemmatizer
lemmatizer = Lemmatizer()


def clean_text(text , classical_model = False):
    """
    Preprocessing Persian news text for Topic Modeling (BERTopic)
    """

    # -------------------------
    # Remove missing values
    # -------------------------
    if not isinstance(text, str):
        return ""

    
    # -------------------------
    # Persian normalization
    # -------------------------
    text = normalizer.normalize(text)


    # -------------------------
    # Remove URLs
    # -------------------------
    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )


    # -------------------------
    # Remove email
    # -------------------------
    text = re.sub(r'\S+@\S+', ' ', text)


    # -------------------------
    # Remove HTML tags
    # -------------------------
    text = re.sub(
        r'<.*?>+',
        '',
        text
    )


    # -------------------------
    # Remove brackets content
    # -------------------------
    text = re.sub(
        r'\[.*?\]',
        '',
        text
    )


    # -------------------------
    # Remove punctuation
    # -------------------------
    text = re.sub(
        r'[^\w\s]',
        '',
        text
    )


    # -------------------------
    # Remove extra spaces
    # -------------------------
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()



    if classical_model:

        # -------------------------
        # Tokenization
        # -------------------------
        tokens = word_tokenize(text)


        # -------------------------
        # Stopword removal
        # -------------------------
        tokens = [
            word 
            for word in tokens
            if word not in persian_stopwords
        ]



        # -------------------------
        # Lemmatization
        # -------------------------
        tokens = [
            lemmatizer.lemmatize(word).split("#")[0]
            for word in tokens
        ]


        # -------------------------
        # Remove very short words
        # -------------------------
        tokens = [
            word 
            for word in tokens
            if len(word) > 1
        ]


        # -------------------------
        # Join tokens
        # -------------------------
        text = " ".join(tokens)


    return text