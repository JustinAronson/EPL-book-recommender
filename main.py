from vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def generate_distance_matrix(vectorizer, dataset):
    vectors = vectorizer.vectorize(dataset)
    np.save("./results/catalog_vectors.npy", vectors)
    return vectors


def get_input_books(vectorizer):
    filename = input("Enter the input book filename")
    with open(filename, "r") as file:
        dataset = np.genfromtxt(file, delimiter=",", dtype=None)

    return vectorizer.vectorize(dataset)


def get_similar_books(catalog_vectors, input_book_vectors):
    similarity = cosine_similarity(input_book_vectors, catalog_vectors)
    similarity_summed = np.sum(similarity, axis=0)

    sorted_indices = np.argsort(similarity_summed)
    most_similar_books = catalog_vectors[sorted_indices[-5:]]

    np.save("./results/sorted_indices.npy", sorted_indices)

    return most_similar_books


with open("./datasets/transactions.csv", "r") as file:
    dataset = np.genfromtxt(file, delimiter=",", dtype=None)

vectorizer = Vectorizer("BAAI/bge-large-en-v1.5")
catalog_vectors = generate_distance_matrix(vectorizer, dataset)

get_input_books(vectorizer)
get_similar_books(catalog_vectors)
