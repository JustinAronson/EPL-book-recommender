from vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random
from collections import defaultdict

def generate_distance_matrix(vectorizer, dataset):
    vectors = vectorizer.vectorize(dataset)
    np.save("./results/catalog_vectors_unique.npy", vectors)
    return vectors

def get_input_books(vectorizer, dataset):
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

        _, indices = np.unique(dataset[:, 1], axis=0, return_index=True)
        dataset = dataset[indices]
        dataset_title_author = dataset[:, 1:3]
        dataset = np.hstack((dataset_title_author, dataset[:, 7:9], dataset[:, -1:]))
        return vectorizer.vectorize(dataset)

def apply_race_ethnicity_parity(similarity_matrix, race_ethnicity_column, parity_factor=1.2):
    for i, race_ethnicity in enumerate(race_ethnicity_column):
        if race_ethnicity != 'White' and race_ethnicity != 'Unknown':
            similarity_matrix[:, i] *= parity_factor

    return similarity_matrix

def get_similar_books(dataset, catalog_vectors, input_book_vectors, parity_factor=1.2):
    # print(catalog_vectors.shape)
    cumulative_similarity = np.zeros(
        (input_book_vectors.shape[0], catalog_vectors.shape[0])
    )

    for column in range(catalog_vectors.shape[2]):
        # print(input_book_vectors[:, :, column].shape, catalog_vectors.shape)
        similarity = cosine_similarity(
            input_book_vectors[:, :, column], catalog_vectors[:, :, column]
        )
        cumulative_similarity += similarity

    race_ethnicity_column = dataset[:, -1]
    adjusted_similarity = apply_race_ethnicity_parity(cumulative_similarity, race_ethnicity_column, parity_factor)

    similarity_summed = np.sum(adjusted_similarity, axis=0)
    # print(similarity_summed.shape)
    
    sorted_indices = np.argsort(similarity_summed)
    most_similar_books = dataset[sorted_indices[-7:]]
    # print(sorted_indices[-7:])
    # print(most_similar_books)

    np.save("./results/sorted_indices.npy", sorted_indices)

    return most_similar_books

def get_dataset(vectorizer):
    with open("./datasets/transactions_author_info_cleaned.csv", "r") as file:
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
        _, indices = np.unique(dataset[:, 1], axis=0, return_index=True)
        dataset = dataset[indices]
        print("unique dataset shape", dataset.shape)
        dataset_title_author = dataset[:, 1:3]
        dataset = np.hstack((dataset_title_author, dataset[:, 7:9], dataset[:, -1:]))
        print(dataset.shape)
    try:
        catalog_vectors = np.load("./results/catalog_vectors_unique.npy")
    except:
        catalog_vectors = generate_distance_matrix(vectorizer, dataset)

    return dataset, catalog_vectors


def test_parity(num_trials, dataset, catalog_vectors):
    parity_dict = defaultdict(int)
    for trial in range(num_trials):
        input_vectors = []
        for i in range(3):
            index = random.randint(0, len(catalog_vectors) - 1)
            input_vectors.append(catalog_vectors[index])
        
        input_vectors = np.array(input_vectors)
        trial_res = get_similar_books(dataset, catalog_vectors, input_vectors, parity_factor=1.2)
        for result in trial_res:
            parity_dict[result[-1]] += 1
    
    print(parity_dict)
    s = sum(parity_dict.values())
    for key, value in parity_dict.items():
        print(key, value/s)

    return parity_dict

def catalog_race_ethnicity_distribution(dataset):
    race_counts = defaultdict(int)

    total_samples = len(dataset)
    for race_ethnicity in dataset[:, -1]:
        race_counts[race_ethnicity] += 1

    # Calculate percentage distribution
    race_percentage = {}
    for race, count in race_counts.items():
        race_percentage[race] = (count / total_samples) * 100

    return race_percentage

vectorizer = Vectorizer("BAAI/bge-large-en-v1.5")
dataset, catalog_vectors = get_dataset(vectorizer)
input_vectors = get_input_books(vectorizer)
most_similar_books = get_similar_books(dataset, catalog_vectors, input_vectors)


# test_parity(50, dataset, catalog_vectors)

# race_distribution = catalog_race_ethnicity_distribution(dataset)

# Print the percentage distribution from the overall catalog
# print("Printing data for the overall catalog:")
# for race, percentage in race_distribution.items():
#     print(f"{race}: {percentage:.2f}%")