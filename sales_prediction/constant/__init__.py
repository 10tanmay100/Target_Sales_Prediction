
import os
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

os.chdir(r"E:\Ivy-Professional-School\project\target sales prediction\Target_Sales_Prediction")
ROOT_DIR=os.getcwd()
CURRENT_TIME_STAMP = get_current_time_stamp()
CONFIG_FILE_PATH="E://Ivy-Professional-School//project//target sales prediction//Target_Sales_Prediction//config//config.yaml"
# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
# Data Ingestion
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"

PATH_READ_LATEST_INGESTION_DATA="E:\\Ivy-Professional-School\\project\\target sales prediction\\Target_Sales_Prediction\\sales_prediction\\artifact\\data_ingestion"
ARTIFACT_DIRECTORY="E:\\Ivy-Professional-School\\project\\target sales prediction\\Target_Sales_Prediction\\sales_prediction"

