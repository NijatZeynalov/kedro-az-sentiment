"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from .nodes import preprocess_sentiment_data, train_val_split
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_sentiment_data,
                inputs=["az_sent_train", "az_sent_test"],
                outputs="tokenized_data",
                name="preprocess_sentiment_data_node"
            ),
            node(
                func=train_val_split,
                inputs=["tokenized_data", "params:data_params"],
                outputs=["train_dataset", "val_dataset"],
                name="train_val_split_node",
            ),
        ]
    )
