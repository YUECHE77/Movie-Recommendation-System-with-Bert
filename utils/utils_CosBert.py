import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm


def get_inputs(csv_file, tokenizer):
    """Process the overviews and genres, so that we can pass them into the model"""

    # Read in the csv file
    df = pd.read_csv(csv_file, keep_default_na=False)
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
    encoded_genres = torch.tensor(encoded_genres, dtype=torch.float32)

    genre_name_to_index = {name: idx for idx, name in enumerate(mlb.classes_)}
    print("Manually Decoded Genres Dictionary: ", genre_name_to_index)

    return input_ids, attention_mask, encoded_genres


def load_dataset(input_ids, attention_mask, encoded_genres, train_ratio=0.8, batch_size=16):
    dataset = TensorDataset(input_ids, attention_mask, encoded_genres)

    train_num = int(train_ratio * len(dataset))
    eval_num = len(dataset) - train_num

    train_dataset, eval_dataset = random_split(dataset, [train_num, eval_num])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader


def evaluate_model(model, data_loader, device, top_k=3):
    total_samples = 0
    top_k_matches = 0

    eval_iterator = tqdm(data_loader, desc='Evaluating', leave=False)

    for data in eval_iterator:
        input_ids, attention_mask, encoded_genres = data
        input_ids, attention_mask, encoded_genres = input_ids.to(device), attention_mask.to(device), encoded_genres.to(device)

        outputs = model(input_ids, attention_mask)
        prob = torch.softmax(outputs, 1)

        top_k_prob, top_k_pred = torch.topk(prob, k=top_k, dim=1)

        for i in range(encoded_genres.size(0)):
            ground_truth = encoded_genres[i].bool()

            pred = torch.zeros_like(encoded_genres[i], dtype=torch.bool)
            pred = pred.scatter(dim=0, index=top_k_pred[i], src=1)

            if (pred & ground_truth).sum() > 0:
                top_k_matches += 1

        total_samples += encoded_genres.size(0)

    accuracy = top_k_matches / total_samples

    return accuracy
