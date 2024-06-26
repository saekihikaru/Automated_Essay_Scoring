import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
# from gensim.models import Word2Vec
from abc import ABC, abstractmethod
import time

class EmbBase(ABC):
    
    def init(self):
        pass


    def embed(self, texts):
        pass

class USEEmbedder(EmbBase):
    def __init__(self, saved_model_path="./use_model"):
        print("Loading USE model...")
        start_time = time.time()
        # TensorFlow SavedModel形式からモデルをロード
        self.model = tf.saved_model.load(saved_model_path)
        elapsed_time = time.time() - start_time
        print(f"Model loaded in {elapsed_time:.2f} seconds")

    def embed(self, texts):
        print("Embedding texts using USE...")
        embeddings = self.model(texts).numpy()
        print("USE embeddings computed.")
        return embeddings

class Word2VecEmbedder(EmbBase):
    def __init__(self, model_path=None, sentences=None, size=100, window=5, min_count=1, workers=4):
        print("Initializing Word2Vec model...")
        if model_path:
            self.model = Word2Vec.load(model_path)
        else:
            self.model = Word2Vec(sentences, vector_size=size, window=window, min_count=min_count, workers=workers)
        print("Word2Vec model initialized.")

    def embed(self, texts):
        print("Embedding texts using Word2Vec...")
        embeddings = np.array([np.mean([self.model.wv[word] for word in text.split() if word in self.model.wv] or [np.zeros(self.model.vector_size)], axis=0) for text in texts])
        print("Word2Vec embeddings computed.")
        return embeddings

# Example usage:
if __name__ == "__main__":
    # Example texts
    # texts = ["This is a sentence.", "This is another sentence."]
    train = pd.read_csv("datasets/train1000.csv")
    test = pd.read_csv("datasets/test.csv")
    # read_csv(file_path, header=None)
    texts = train['full_text']
    print("Texts to be embedded:", texts)

    # # 事前にダウンロード
    # hub_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # print("Downloading and saving the USE model...")
    # print("Loading USE model...")
    # start_time = time.time()
    # print(start_time)
    # use_model = hub.load(hub_model_url)
    # elapsed_time = time.time() - start_time
    # print(f"Model loaded in {elapsed_time:.2f} seconds")

    # # 保存先ディレクトリを指定
    # saved_model_path = "./use_model"
    # # モデルをTensorFlow SavedModel形式で保存
    # tf.saved_model.save(use_model, saved_model_path)
    # print("Model saved locally at", saved_model_path)
    # exit()

    # USE Embedding
    print("Starting USE embedding...")
    use_embedder = USEEmbedder()

    # テキストの埋め込み
    print("Embedding training texts using USE...")
    train_embeddings = use_embedder.embed(train['full_text'])
    train_embeddings = pd.DataFrame(train_embeddings)
    train = pd.concat([train, train_embeddings], axis=1)

    print("Embedding test texts using USE...")
    test_embeddings = use_embedder.embed(test['full_text'])
    test_embeddings = pd.DataFrame(test_embeddings)
    test = pd.concat([test, test_embeddings], axis=1)

    # データの情報表示
    print("Training data info:")
    print(train.info())
    print("Training data head:")
    print(train.head(10))

    print("Test data info:")
    print(test.info())
    print("Test data head:")
    print(test.head(10))

    # # Word2Vec Embedding
    # print("Starting Word2Vec embedding...")
    # sentences = [text.split() for text in texts]
    # word2vec_embedder = Word2VecEmbedder(sentences=sentences)
    # word2vec_embeddings = word2vec_embedder.embed(texts)
    # print("Word2Vec Embeddings:", word2vec_embeddings)


    train.to_csv('datasets/train_embed1000_use.csv', index=False)
    test.to_csv('datasets/test_embed_use.csv', index=False)
