import tensorflow as tf
from pathlib import Path
import mlflow
import pandas as pd
import mlflow.keras
from urllib.parse import urlparse
from reviewAnalysis.entity.config_entity import EvaluationConfig
from sklearn.model_selection import train_test_split

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _prepare_data(self):
        train_df = pd.read_csv(self.config.training_data)
    
        # Assuming the CSV contains 'text' column for texts and 'label' column for labels
        self.train_texts = train_df['Review'].tolist()
        self.train_labels = train_df['Rating'].tolist()
    
        # Split training data into training and validation sets
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        self.train_texts, self.train_labels, test_size=0.2, random_state=42)
    
        # Assign the split data to instance variables
        self.train_texts = train_texts
        self.valid_texts = valid_texts
        self.train_labels = train_labels
        self.valid_labels = valid_labels



    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._prepare_data()
        self.score = self.model.evaluate(self.valid_texts, self.valid_labels)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        # save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.keras.log_model(self.model, "model", registered_model_name="LSTM")
            else:
                mlflow.keras.log_model(self.model, "model")
