from hazm import stopwords_list

stop_words = stopwords_list()
custom_stopwords = [
    "گفت",
    "کرد",
    "شد",
    "می‌شود",
    "خواهد",
    "داشت",
    "کرده",
    "گزارش",
    "خبرنگار",
    "ایسنا",
]

stop_words = stop_words + custom_stopwords
