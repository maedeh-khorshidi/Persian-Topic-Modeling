import torch
from transformers import AutoTokenizer, AutoModel

class ParsBERTEncoder:

    def __init__(
        self,
        model_name="HooshvareLab/bert-base-parsbert-uncased",
        batch_size=32,
        max_length=512
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

        self.model.eval()

        self.batch_size = batch_size

        self.max_length = max_length

        print(f"Using device: {self.device}")


    def tokenize(self, texts):

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return encoded_input
    
    
    def forward_pass(self, encoded_input):

        encoded_input = {
            key: value.to(self.device)
            for key, value in encoded_input.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **encoded_input
            )

        return outputs
    
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = (
        model_output.last_hidden_state
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return (
            torch.sum(
                token_embeddings * mask,
                dim=1
            )
            /
            torch.clamp(
                mask.sum(dim=1),
                min=1e-9
            )
        )
    
    
    def encode(self, texts):

        all_embeddings = []

        for i in range(
            0,
            len(texts),
            self.batch_size
        ):

            batch = texts[
                i:i+self.batch_size
            ]

            encoded_input = self.tokenize(batch)

            outputs = self.forward_pass(
                encoded_input
            )

            embeddings = self.mean_pooling(
                outputs,
                encoded_input["attention_mask"].to(self.device)
            )

            all_embeddings.append(
                embeddings.cpu()
            )

        all_embeddings = torch.cat(
            all_embeddings,
            dim=0
        )

        return all_embeddings.numpy()