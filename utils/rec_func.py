import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_movies_with_index(movies_, embeddings, movie_index, k=5):
    """
    Input a movie index, return recommendations
    :param movies_: Your Dataset(Movie Database) -> DataFrame
    :param embeddings: Words Embeddings
    :param movie_index: Index of movie
    :param k: Number of recommendations
    :return: None -> result is printed
    """
    if movie_index >= movies_.shape[0]:
        print(f'Invalid movie index {movie_index}!')
        return

    k = 5 if k <= 0 else k

    movie_emb = embeddings[movie_index]  # find the corresponding embedding vector first
    cos_simi = cosine_similarity([movie_emb], embeddings)[0]  # compute the similarity to all movies

    top_indices = np.argsort(-cos_simi)[1:k + 1]  # because the largest cosine similarity must be with itself
    rec_movies = movies_.iloc[top_indices]  # get the recommendation movies -> DataFrame

    print(f'The movie {movie_index}: {movies_.title.iloc[movie_index]} \t Genres: {movies_.AllGenres.iloc[movie_index]}\n')
    print(f'Top {k} recommendations:')

    for index, movie in rec_movies.iterrows():
        print('{:<6} {:<35} {}'.format(f'{index}.', f'{movie["title"]}', f'Genres: {movie["AllGenres"]}'))


def recommend_movies_with_titles(titles, movies_, embeddings, model_, k=5):
    """
    Input a list of movie(s), return recommendations
    :param titles: Should be a list -> Title(s) of movie(s)
    :param movies_: Your Dataset(Movie Database) -> DataFrame
    :param embeddings: Words Embeddings
    :param model_: The model
    :param k: Number of recommendations
    :return: None -> result is printed
    """
    if not titles:
        print('Enter at least ONE movie title!')
        return

    k = 5 if k <= 0 else k

    movies_input = movies_[movies_['title'].isin(titles)]  # All input movies

    # If None of the movie(s) title(s) that the user enters is in our dataset
    # The only useful information we have is those titles -> encode those titles -> get embeddings
    if movies_input.empty:
        print('Those movies are not in our Database currently, but we can recommend you the following: \n')

        movies_remaining = movies_  # Every movie in the dataset is the "remaining movie"

        user_inputs = ','.join(titles)
        movies_input_emb = model_.encode(user_inputs)
        combined_emb = np.array(movies_input_emb)  # Why use this name? -> Just for convenience

    # If the user enters valid titles, then just get the embeddings for the movies
    else:
        movies_remaining = movies_[~movies_['title'].isin(titles)]  # All remaining movies

        movies_input_emb = embeddings[movies_input.index]  # embeddings is np.array
        combined_emb = np.mean(movies_input_emb, axis=0)  # Compute the mean

    movies_remaining_emb = embeddings[movies_remaining.index]  # Embedding of remaining movies

    cos_simi = cosine_similarity([combined_emb], movies_remaining_emb)[0]

    top_indices = np.argsort(-cos_simi)[:k]
    recommend_movies = movies_remaining.iloc[top_indices]

    print(f'Top {k} recommendations:')
    for index, movie in recommend_movies.iterrows():
        print('{:<6} {:<35} {}'.format(f'{index}.', f'{movie["title"]}', f'Genres: {movie["AllGenres"]}'))


def recommend_movies_with_history(user_id, ratings_, movies_, embeddings, model_, k=5, m=3):
    """
    Input a user ID, return recommendations
    :param user_id: The user id
    :param ratings_: All users watching history -> DataFrame
    :param movies_: Your Dataset(Movie Database) -> DataFrame
    :param embeddings: Words Embeddings
    :param model_: The model
    :param k: Number of recommendations
    :param m: Number of watching history that used
    :return: None -> result is printed
    """
    user_list = ratings_['userId'].unique().tolist()
    if user_id not in user_list:
        print(f'Cannot find the user with id: {user_id}')
        return

    user_history = ratings_[ratings_['userId'] == user_id]  # get the user's history (all)

    m = 3 if m <= 0 else m
    m = min(m, len(user_history))

    # Sort movies with the highest user ratings and recently watched
    sorted_history = user_history.sort_values(by=['rating', 'timestamp'])
    sorted_m_history = sorted_history.iloc[:m]  # select the first m movies

    user_history_id = sorted_m_history['movieId'].tolist()

    history_titles = movies_[movies_['id'].isin(user_history_id)]
    history_titles = history_titles['title'].tolist()

    recommend_movies_with_titles(history_titles, movies_, embeddings, model_=model_, k=k)


def genres_to_movies(genres_, model_, movies_, embeddings, k=5):
    """
    From genres to movies -> This function is additional
    :param genres_: Should be a list -> The genres
    :param model_: The model
    :param movies_: Your Dataset(Movie Database) -> DataFrame
    :param embeddings: Words Embeddings
    :param k: Number of recommendations
    :return: Movie(s) title(s) in a list
    """
    if not genres_:
        print('Enter at least ONE genre!')
        return

    k = 5 if k <= 0 else k

    all_genres = ','.join(genres_)
    all_genres_emb = model_.encode(all_genres)
    all_genres_emb = np.array(all_genres_emb)

    cos_simi = cosine_similarity([all_genres_emb], embeddings)[0]

    top_indices = np.argsort(-cos_simi)[:k]
    recommend_movies = movies_.iloc[top_indices]

    print(f'Top {k} recommendations:')
    for index, movie in recommend_movies.iterrows():
        print('{:<6} {:<35} {}'.format(f'{index}.', f'{movie["title"]}', f'Genres: {movie["AllGenres"]}'))

    return recommend_movies['title'].tolist()
