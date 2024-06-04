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
        columns = []
        for i in range(dataset.shape[1]):
            print('col ', i, ' done')
            columns.append(self.encode_unique(dataset[:, i]))
        return np.stack(columns, axis=2)
