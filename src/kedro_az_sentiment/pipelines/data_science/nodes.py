"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict

# Constants
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512


class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        return item


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
        'test_labels': test_labels,
        'label_encoder': label_encoder
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
    train_dataset = SentimentDataset(x_train, mask_train, y_train)
    val_dataset = SentimentDataset(x_val, mask_val, y_val)

    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset, parameters: Dict):
    """Trains the BERT model.
    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        parameters: Dictionary containing training parameters.
    Returns:
        Trained model.
    """
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(set(train_dataset.labels.numpy())))

    training_args = TrainingArguments(
        output_dir=parameters["output_dir"],
        num_train_epochs=parameters["num_train_epochs"],
        per_device_train_batch_size=parameters["batch_size"],
        per_device_eval_batch_size=parameters["batch_size"],
        warmup_steps=parameters["warmup_steps"],
        weight_decay=parameters["weight_decay"],
        logging_dir=parameters["logging_dir"],
        logging_steps=parameters["logging_steps"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    data_collator = DataCollatorWithPadding(tokenizer=BertTokenizer.from_pretrained(MODEL_NAME))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    return model


def evaluate_model(model, val_dataset):
    """Evaluates the BERT model.
    Args:
        model: Trained model.
        val_dataset: Validation dataset.
    Returns:
        Evaluation metrics.
    """
    trainer = Trainer(model=model)
    results = trainer.evaluate(eval_dataset=val_dataset)
    return results


def compute_metrics(eval_pred):
    """Computes metrics for evaluation.
    Args:
        eval_pred: Evaluation predictions.
    Returns:
        Dictionary with accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
