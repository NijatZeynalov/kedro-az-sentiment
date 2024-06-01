"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from typing import Dict
from sklearn.preprocessing import LabelEncoder

# Constants
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512


def preprocess_sentiment_data(az_sent_train: pd.DataFrame, az_sent_test: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Preprocesses the sentiment analysis data.
    Args:
        az_sent_train: Raw training data with 'text' and 'labels' columns.
        az_sent_test: Raw test data with 'text' and 'labels' columns.
    Returns:
        Preprocessed data ready for BERT model.
    """
    # Extract text and labels
    train_texts = az_sent_train['text'].values
    train_labels = az_sent_train['labels'].values
    test_texts = az_sent_test['text'].values
    test_labels = az_sent_test['labels'].values

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize texts
    train_tokenized = tokenizer(
        list(train_texts),
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    test_tokenized = tokenizer(
        list(test_texts),
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    return {
        'train_input_ids': train_tokenized['input_ids'],
        'train_attention_mask': train_tokenized['attention_mask'],
        'train_labels': train_labels,
        'test_input_ids': test_tokenized['input_ids'],
        'test_attention_mask': test_tokenized['attention_mask'],
        'test_labels': test_labels
    }


def train_val_split(tokenized_data: Dict[str, torch.Tensor], parameters: Dict):
    """Splits the data into training and validation sets.
    Args:
        tokenized_data: Tokenized text data.
        parameters: Dictionary containing 'val_size' and 'random_state'.
    Returns:
        Training and validation sets.
    """
    input_ids = tokenized_data['train_input_ids']
    attention_mask = tokenized_data['train_attention_mask']
    labels = tokenized_data['train_labels']

    # Split the data
    x_train, x_val, mask_train, mask_val, y_train, y_val = train_test_split(
        input_ids, attention_mask, labels, test_size=parameters["val_size"], random_state=parameters["random_state"]
    )

    # Create torch datasets
    train_dataset = torch.utils.data.TensorDataset(x_train, mask_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, mask_val, y_val)

    return train_dataset, val_dataset
