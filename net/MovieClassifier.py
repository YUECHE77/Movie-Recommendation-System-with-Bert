import torch.nn as nn


class MovieClassifier(nn.Module):
    def __init__(self, num_genres, bert_model, dropout_rate=0.3):
        super(MovieClassifier, self).__init__()

        self.bert = bert_model
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_genres)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        dropout = self.dropout(pooler_output)
        results = self.classifier(dropout)

        return results
