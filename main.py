from vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def generate_distance_matrix(vectorizer, dataset):
    vectors = vectorizer.vectorize(dataset)
    np.save("./results/catalog_vectors.npy", vectors)
    return vectors


def get_input_books(vectorizer):
    # filename = input("Enter the input book filename")
    filename = "datasets/inputs.csv"
    with open(filename, "r") as file:
        dataset = np.genfromtxt(
            file,
            delimiter=",",
            dtype=str,
            skip_header=0,
            filling_values="",
            invalid_raise=False,
        )

    return vectorizer.vectorize(dataset)


def get_similar_books(catalog_vectors, input_book_vectors):
    cumulative_similarity = np.zeros(
        (input_book_vectors.shape[0], catalog_vectors.shape[0])
    )

    for column in range(input_book_vectors.shape[2]):
        print(input_book_vectors[:, :, column].shape, catalog_vectors.shape)
        similarity = cosine_similarity(
            input_book_vectors[:, :, column], catalog_vectors[:, :, column]
        )
        cumulative_similarity = np.sum((cumulative_similarity, similarity), axis=0)

    similarity_summed = np.sum(cumulative_similarity, axis=0)
    print(similarity_summed.shape)

    sorted_indices = np.argsort(similarity_summed)
    most_similar_books = catalog_vectors[sorted_indices[-5:]]
    print(sorted_indices[-5:])

    np.save("./results/sorted_indices.npy", sorted_indices)

    return most_similar_books


def get_dataset(vectorizer):
    try:
        catalog_vectors = np.load("./results/catalog_vectors.npy")
    except:
        with open("./datasets/transactions_cleaned.csv", "r") as file:
            dataset = np.genfromtxt(
                file,
                delimiter=",",
                dtype=str,
                skip_header=1,
                filling_values="",
                invalid_raise=False,
            )
            print("dataset loaded")
            print("dataset shape:")
            print(dataset.shape)

        catalog_vectors = generate_distance_matrix(vectorizer, dataset)
    return catalog_vectors


vectorizer = Vectorizer("BAAI/bge-large-en-v1.5")
catalog_vectors = get_dataset(vectorizer)
input_vectors = get_input_books(vectorizer)
most_similar_books = get_similar_books(catalog_vectors, input_vectors)
print(most_similar_books)
