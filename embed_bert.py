import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from abc import ABC, abstractmethod
import time

class EmbBase(ABC):
    
    def init(self):
        pass

    @abstractmethod
    def embed(self, texts):
        pass

class DistilBertEmbedder(EmbBase):
    def __init__(self, model_name="distilbert-base-uncased"):
        print("Loading DistilBERT model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        print(f"Model loaded on {self.device}")

    def embed(self, texts, batch_size=8):
        print("Embedding texts using DistilBERT...")
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
        print("DistilBERT embeddings computed.")
        return embeddings

if __name__ == "__main__":
    print("Checking if GPU is available...")
    print("Num GPUs Available: ", torch.cuda.device_count())
    print("Torch is using GPU: ", torch.cuda.is_available())

    # Load data
    train = pd.read_csv("datasets/train1000.csv")
    test = pd.read_csv("datasets/test.csv")
    texts = train['full_text'].tolist()
    print("Texts to be embedded:", texts[:10])  # Display first 10 texts for reference

    # DistilBERT Embedding
    print("Starting DistilBERT embedding...")
    distilbert_embedder = DistilBertEmbedder()

    # Embed training texts
    print("Embedding training texts using DistilBERT...")
    train_embeddings = distilbert_embedder.embed(train['full_text'].tolist())
    train_embeddings = pd.DataFrame(train_embeddings)
    train = pd.concat([train, train_embeddings], axis=1)

    # Embed test texts
    print("Embedding test texts using DistilBERT...")
    test_embeddings = distilbert_embedder.embed(test['full_text'].tolist())
    test_embeddings = pd.DataFrame(test_embeddings)
    test = pd.concat([test, test_embeddings], axis=1)

    # Display data info
    print("Training data info:")
    print(train.info())
    print("Training data head:")
    print(train.head(10))

    print("Test data info:")
    print(test.info())
    print("Test data head:")
    print(test.head(10))

    # Save embeddings to CSV
    train.to_csv('datasets/train_embed1000_bert.csv', index=False)
    test.to_csv('datasets/test_embed_bert.csv', index=False)
