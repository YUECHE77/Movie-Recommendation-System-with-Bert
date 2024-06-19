from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split, TensorDataset
import utils.utils_CosBert as CosBert
from net.MovieClassifier import MovieClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os
import glob


if __name__ == '__main__':
    # ----------------------------------------------------#
    #   overview_genre_path: The overviews and genres dataset
    # ----------------------------------------------------#
    overview_genre_path = 'dataset/movies_overview_genre.csv'
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
    lr = 1e-4
    output_path = 'logs/Sentence_Bert/training_nli_distilbert-model'

    pretrained_model = 'bert-base-uncased'
    # ----------------------------------------------------#
    #   Load the tokenizer
    #   Load the data
    # ----------------------------------------------------#
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    input_ids, attention_mask, encoded_genres = CosBert.get_inputs(overview_genre_path, tokenizer)
    train_loader, eval_loader = CosBert.load_dataset(input_ids, attention_mask, encoded_genres, batch_size=32)
    # ----------------------------------------------------#
    #   Load the model and put it on GPU
    # ----------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_model = BertModel.from_pretrained(pretrained_model)

    model = MovieClassifier(num_genres=encoded_genres.size(1), bert_model=bert_model)
    model.to(device)
    # ----------------------------------------------------#
    #   Set up optimizer and loss function
    # ----------------------------------------------------#
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    # ----------------------------------------------------#
    #   Start Training
    # ----------------------------------------------------#
    print('\nStart Training!!!\n')

    for epoch in range(epoch_num):
        model.train()
        total_batches = len(train_loader)

        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for data in train_loader:
                input_ids, attention_mask, encoded_genres = data
                input_ids, attention_mask, encoded_genres = input_ids.to(device), attention_mask.to(device), encoded_genres.to(device)

                output = model(input_ids, attention_mask)
                loss = loss_func(output, encoded_genres)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                with torch.no_grad():
                    model.eval()

                    accuracy = CosBert.evaluate_model(model, eval_loader, device)
                    print(f'Epoch: {epoch + 1:02d}, Accuracy: {accuracy:.3f}')

                    if (epoch + 1) % 5 == 0:
                        sub_path = int(1000 * accuracy)
                        save_path = f'logs/Costomer_Bert/model_acc_{sub_path}.pth'
                        torch.save(model.state_dict(), save_path)

    print('\nFinished Training!!!\n')
