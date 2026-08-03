from ollama import chat
from config import(
    TOPIC_LABEL_MAX_WORDS,
    OLLAMA_MODEL,
    N_TOPIC_WORDS
)


class TopicLabelGenerator:

    def __init__(
        self,
        topic_model
    ):

        self.topic_model = topic_model

        self.labels = {}

    
    def build_prompt(
        self,
        topic_words
    ):
        """
        Create prompt for Llama.
        """

        words = ", ".join(topic_words)

        prompt = f"""
    شما یک متخصص دسته‌بندی موضوعات خبری فارسی هستید.

    وظیفه:
    برای کلمات کلیدی زیر فقط یک عنوان کوتاه و معنادار برای موضوع تولید کن.

    قوانین:
    - فقط عنوان را برگردان.
    - حداکثر {str(TOPIC_LABEL_MAX_WORDS)} کلمه.
    - هیچ توضیحی ننویس.
    - از نقل قول، شماره یا بولت استفاده نکن.
    - عنوان باید طبیعی و مناسب اخبار فارسی باشد.

    کلمات کلیدی:
    {words}
    """

        return prompt
    

    def generate_label(self, topic_words):

        prompt = self.build_prompt(topic_words)

        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"].strip()