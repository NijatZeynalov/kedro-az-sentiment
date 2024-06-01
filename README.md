# Sentiment Analysis in Azerbaijani with BERT using Kedro

This project implements a sentiment analysis pipeline for Azerbaijani language using the BERT model, facilitated by the Kedro framework. The data processing, model training, and evaluation are streamlined into a reproducible workflow, ensuring ease of experimentation and deployment.

# Pipeline Visualization

![kedro-pipeline](https://github.com/NijatZeynalov/kedro-az-sentiment/assets/31247506/f95b3f67-fb9e-4873-b187-b2a8ae917c2a)

# Installation
Clone the repository:

```
git clone https://github.com/nijatzeynalov/kedro-az-sentiment.git
cd kedro-az-sentiment
```

Create a virtual environment and activate it:

```
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```
pip install -r src/requirements.txt
```

Configuration Data: 

Place  az_sent_train.csv and az_sent_test.csv files in the data/01_raw directory. The dataset sourced from LocalDoc/sentiments_dataset_azerbaijani.

Configuration Files:

conf/base/catalog.yml

conf/base/data_processing.yml

conf/base/train_params.yml

conf/base/logging.yml

