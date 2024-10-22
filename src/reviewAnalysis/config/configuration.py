from reviewAnalysis.constants import *
from reviewAnalysis.utils.common import read_yaml, create_directories
# from reviewAnalysis.entity.config_entity import DataIngestionConfig
# from reviewAnalysis.entity.config_entity import DataValidationConfig
# from reviewAnalysis.entity.config_entity import DataTransformationConfig
from reviewAnalysis.entity.config_entity import (DataIngestionConfig,
                                                     DataValidationConfig,
                                                     DataTransformationConfig,
                                                     ModelTrainerConfig)
from reviewAnalysis.entity.config_entity import EvaluationConfig



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            root2_dir=config.root2_dir,
            data_path=config.data_path,
            max_words = params.max_words,
            max_len = params.max_len,
            batch_size = params.batch_size,
            epochs = params.epochs,
            validation_split = params.validation_split
            
        )

        return model_trainer_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="src/reviewAnalysis/models/model.h5",
            training_data="artifacts/data_ingestion/flipkart.csv",
            mlflow_uri="https://dagshub.com/subhayudutta/Sentiment-Analysis-of-Flipkart-Reviews.mlflow",
            all_params=self.params
        )
        return eval_config