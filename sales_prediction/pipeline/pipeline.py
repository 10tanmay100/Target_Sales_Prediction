from mimetypes import init
from sales_prediction.config.configuration import Configuration
from sales_prediction.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig
from sales_prediction.logger import logging
from sales_prediction.component.data_ingestion import DataIngestion
from sales_prediction.component.data_validation import DataValidation
from sales_prediction.component.data_transformation import DataTransformation
from sales_prediction.component.model_trainer import ModelTrainer
from sales_prediction.entity.artifact_entity import *
from sales_prediction.exception import sales_project_exception
from sales_prediction.constant import *
from sales_prediction.util.util import *

class Pipeline:
    def __init__(self, config: Configuration ) -> None:
        self.config=config()
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            initiate=data_ingestion.initiate_data_ingestion()
            return DataIngestionArtifact(raw_data_file1_path=initiate.raw_data_file1_path,raw_data_file2_path=initiate.raw_data_file2_path,raw_data_file3_path=initiate.raw_data_file3_path,is_ingested="yes",message="Successfull")
        except Exception as e:
            raise sales_project_exception(e, sys) from e 


    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) \
            -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise sales_project_exception(e, sys) from e


    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise sales_project_exception(e, sys)
    

    def start_model_training(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer=ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_training()
        except Exception as e:
            raise sales_project_exception(e,sys) from e

    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact=self.start_model_training(data_transformation_artifact=data_transformation_artifact)
            return "done"
        except Exception as e:
            raise sales_project_exception(e,sys) from e