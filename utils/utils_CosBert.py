from transformers import BertTokenizer, BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from tqdm import tqdm
import json


def get_inputs(csv_file, tokenizer):
    """Process the overviews and genres, so that we can pass them into the model"""
    df = pd.read_csv(csv_file)
    overviews = df['overview'].tolist()  # convert to a list -> ['overview_1', 'overview_2', ..., 'overview_n']
    genres = df['AllGenres'].tolist()

    def encode_overview(overviews_, tokenizer_):
        """encode overviews"""
        input_ids_ = []
        attention_mask_ = []

        for overview in overviews_:
            encoded = tokenizer_.encode_plus(overview,
                                             add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                             max_length=512,  # 512 is the max length for BERT tokenizer
                                             padding='max_length',
                                             return_attention_mask=True,
                                             truncation=True,
                                             return_tensors='pt')

            # The reason for indexing like the following is that that's how BERT tokenizer returns
            input_ids_.append(encoded['input_ids'][0])
            attention_mask_.append(encoded['attention_mask'][0])

        # Stack them together -> get a long tensor
        input_ids_ = torch.stack(input_ids_)
        attention_mask_ = torch.stack(attention_mask_)

        return input_ids_, attention_mask_

    input_ids, attention_mask = encode_overview(overviews, tokenizer)

    # The genres here is our labels -> each movie might have multiple labels
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(genres)

    genre_name_to_index = {name: idx for idx, name in enumerate(mlb.classes_)}
    print("Manually Decoded Genres Dictionary: ", genre_name_to_index)

    return input_ids, attention_mask, torch.tensor(encoded_genres, dtype=torch.float32)
