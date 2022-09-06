from collections import namedtuple

DataIngestionConfig=namedtuple("DataIngestionConfig",
["raw_data_dir"])

#DataValidationConfig
DataValidationConfig = namedtuple("DataValidationConfig",["artifact_path","business_csv_schema_file_path",
"sales_history_csv_schema_file_path",
"store_details_csv_schema_file_path"])




#datapipelineconfig
TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])