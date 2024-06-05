import logging

import torch
from sentence_transformers import SentenceTransformer, losses, LoggingHandler

import utils.utils_SBert as SBert

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   movies_all_path: The whole movie dataset
    #   movies_small_path: Partial of movie dataset
    #   ratings_for_history_path: The whole users watching history
    #   ratings_for_history_small_path: Partial of watching history
    # ----------------------------------------------------#
    movies_all_path = 'dataset/movies_with_keywords.csv'
    movies_small_path = 'dataset/movies_5000.csv'
    ratings_for_history_path = 'dataset/ratings_for_history.csv'
    ratings_for_history_small_path = 'dataset/ratings_for_history_small.csv'

    # ----------------------------------------------------#
    #   Training parameters
    #   epoch_num       Epoch number
    #   batch_size      Batch size
    #   warm_up         Training warm up
    #   eval_steps      Steps to evaluate the model
    #   lr              Learning rate
    #   output_path     Where do you save your model
    # ----------------------------------------------------#
    epoch_num = 1
    batch_size = 16
    warm_up = 100
    eval_steps = 100
    lr = 1e-6
    output_path = 'logs/Sentence_Bert/training_nli_distilbert-model'

    # ----------------------------------------------------#
    #   Read in the data
    # ----------------------------------------------------#
    movies, movies_5000, ratings_for_history, ratings_for_history_small = SBert.read_in_csv(movies_all_path,
                                                                                            movies_small_path,
                                                                                            ratings_for_history_path,
                                                                                            ratings_for_history_small_path)

    # ----------------------------------------------------#
    #   Generate data points for training and validation
    # ----------------------------------------------------#
    examples_train = SBert.generate_data(movies_=movies, num=100)
    examples_val = SBert.generate_data(movies_=movies, num=30)

    # ----------------------------------------------------#
    #   Load training data and get the evaluator
    # ----------------------------------------------------#
    train_loader = SBert.load_dataset(examples_train, batch_size=batch_size)
    evaluator = SBert.get_evaluator(examples_val, 'evaluator1')

    # ----------------------------------------------------#
    #   Load the model and put it on GPU
    # ----------------------------------------------------#
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer('distilbert-base-nli-mean-tokens').to(device)

    # ----------------------------------------------------#
    #   Set up logging
    # ----------------------------------------------------#
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # ----------------------------------------------------#
    #   The loss function -> call it train_loss
    # ----------------------------------------------------#
    train_loss = losses.CosineSimilarityLoss(model=model)

    # ----------------------------------------------------#
    #   Fit the model
    # ----------------------------------------------------#
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=epoch_num,
        warmup_steps=warm_up,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': lr},
        evaluation_steps=eval_steps,
        output_path=output_path
    )

    print('Finished Training')
