from sentence_transformers import SentenceTransformer
import numpy as np
import pickle


class Vectorizer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode_unique(self, column):
        unique_vals = np.unique(column)
        unique_encodings = self.model.encode(list(unique_vals))

        val_dict = {
            key: encoding for key, encoding in zip(unique_vals, unique_encodings)
        }

        return np.array([val_dict[val] for val in column])

    def vectorize(self, dataset):
        for i in range(dataset.shape[1]):
            dataset[:, i] = self.encode_unique(dataset[:, i])
        return dataset
        return np.stack((pathogenicity_encodings, encodings), axis=2)
