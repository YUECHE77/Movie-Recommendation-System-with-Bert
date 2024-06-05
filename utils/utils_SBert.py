import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from sentence_transformers import InputExample, evaluation
from .dataset import SBert_Dataset


def read_in_csv(movies_all_path, movies_small_path, ratings_for_history_path, ratings_for_history_small_path):
    """
    Especially for processing the data used for Sentence Bert
    :param movies_all_path: The whole movie dataset (Your database)
    :param movies_small_path: Partial of movie dataset
    :param ratings_for_history_path: The whole users watching history
    :param ratings_for_history_small_path: Partial of watching history
    :return: Four DataFrames
    """
    movies = pd.read_csv(movies_all_path, keep_default_na=False, dtype=str)
    movies_5000 = pd.read_csv(movies_small_path, keep_default_na=False, dtype=str)
    ratings_for_history = pd.read_csv(ratings_for_history_path, dtype=str)
    ratings_for_history_small = pd.read_csv(ratings_for_history_small_path, dtype=str)

    ratings_for_history['rating'] = ratings_for_history['rating'].astype(float)
    ratings_for_history['timestamp'] = ratings_for_history['timestamp'].astype(int)
    ratings_for_history_small['rating'] = ratings_for_history_small['rating'].astype(float)
    ratings_for_history_small['timestamp'] = ratings_for_history_small['timestamp'].astype(int)

    return movies, movies_5000, ratings_for_history, ratings_for_history_small


def embed_sentences(movies_, model_):
    """Embed movies titles, genres, and keywords -> Descriptions"""
    descriptions = movies_['title'] + ',' + movies_['AllGenres'] + ',' + movies_['AllKeywords']
    descriptions = descriptions.to_list()
    embeddings = model_.encode(descriptions, show_progress_bar=True)
    embeddings = np.array(embeddings)

    return embeddings


def generate_data(movies_, num=100):
    """Generate data for model training and validation"""
    sample_index = np.random.choice(len(movies_), size=num, replace=False)
    sample_index = np.sort(sample_index)  # must be sorted!!

    def if_similar(movie1_, movie2_):
        """
        The function compute the similarity of two movies:
        Only consider genres, because there are way too many unique keywords
        """
        genres1 = set(movie1_['AllGenres'].strip().split(','))
        genres2 = set(movie2_['AllGenres'].strip().split(','))

        min_len = min(len(genres1), len(genres2))  # the minimum of two lengths
        if min_len == 0:
            return 0.1  # Just a ramdom thought

        common_genres = genres1 & genres2  # Find the overlapped genres
        common_genres_num = len(common_genres)

        similarity_ = (common_genres_num / float(min_len))

        return round(similarity_, 2)

    examples = []
    for i in sample_index:
        movie1 = movies_.iloc[i]

        for j in sample_index:
            if j > i:
                movie2 = movies_.iloc[j]
                similarity = if_similar(movie1, movie2)

                examples.append(
                    InputExample(texts=[
                        movie1['title'] + ',' + movie1['AllGenres'] + ',' + movie1['AllKeywords'],
                        movie2['title'] + ',' + movie2['AllGenres'] + ',' + movie2['AllKeywords']
                    ], label=similarity)
                )

    return examples


def load_dataset(examples, batch_size):
    """Load the dataset -> get train loader"""
    my_dataset = SBert_Dataset(examples)
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_datapoints_num(examples):
    print(f'There are {len(examples)} data points in total')


def get_evaluator(examples_val, name):
    """Return an evaluator -> you can actually have multiple evaluators during training, but I never try that"""
    return evaluation.EmbeddingSimilarityEvaluator.from_input_examples(examples_val, name=name)


def print_model_summary(model):
    """Print the model structure"""
    print("Model summary:\n")

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        total_params += param
        print(f"{name}: {param} total parameters")

    print(f"\nTotal trainable parameters: {total_params}")
