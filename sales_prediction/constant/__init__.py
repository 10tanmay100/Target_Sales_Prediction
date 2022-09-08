
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

# Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_BUSINESS_CSV_SCHEMA_FILE_NAME_KEY = "business_csv_schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_ARTIFACT_DIR_NAME="data_validation"
DATA_VALIDATION_BUSINESS_CSV_REPORT_FILE_NAME_KEY = "business_csv_report_file_name"
DATA_VALIDATION_BUSINESS_CSV_REPORT_PAGE_FILE_NAME_KEY = "business_csv_report_page_file_name"

DATA_VALIDATION_SALES_HISTORY_CSV_SCHEMA_FILE_NAME_KEY = "sales_history_csv_schema_file_name"
DATA_VALIDATION_SALES_HISTORY_CSV_REPORT_FILE_NAME_KEY = "sales_history_csv_report_file_name"
DATA_VALIDATION_SALES_HISTORY_CSV_REPORT_PAGE_FILE_NAME_KEY = "sales_history_csv_report_page_file_name"


DATA_VALIDATION_STORE_DETAILS_CSV_SCHEMA_FILE_NAME_KEY = "store_details_csv_schema_file_name"
DATA_VALIDATION_STORE_DETAILS_CSV_REPORT_FILE_NAME_KEY = "store_details_csv_report_file_name"
DATA_VALIDATION_STORE_DETAILS_CSV_REPORT_PAGE_FILE_NAME_KEY = "store_details_csv_report_page_file_name"


DATA_VALIDATION_BUSINESS_FOLDER_NAME="Business_csv_validation"
DATA_VALIDATION_SALES_HISTORY_FOLDER_NAME="Sales_history_csv_validation"
DATA_VALIDATION_STORE_DETAILS_FOLDER_NAME="Store_details_csv_validation"


DATA_INGESTION_LATEST_DIR="E:\\Ivy-Professional-School\\project\\target sales prediction\\Target_Sales_Prediction\\sales_prediction\\artifact\\data_ingestion"


BUSINESS_CSV_SCHEMA_FILE_PATH="E:\\Ivy-Professional-School\\project\\target sales prediction\\Target_Sales_Prediction\\config\\business_csv_schema.yaml"

SALES_HISTORY_SCHEMA_FILE_PATH="E:\\Ivy-Professional-School\\project\\target sales prediction\\Target_Sales_Prediction\\config\\sales_history_csv_schema.yaml"

STORE_DETAILS_SCHEMA_FILE_PATH="E:\\Ivy-Professional-School\\project\\target sales prediction\\Target_Sales_Prediction\\config\\store_details_schema.yaml"


# Data Transformation related variables
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_VALIDATE_DIR_NAME_KEY = "transformed_validate_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = "preprocessed_object_file_name"

