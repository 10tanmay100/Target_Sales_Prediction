from collections import namedtuple

DataIngestionConfig=namedtuple("DataIngestionConfig",
["raw_data_dir"])

#DataValidationConfig
DataValidationConfig = namedtuple("DataValidationConfig",["artifact_path","business_csv_schema_file_path",
"sales_history_csv_schema_file_path",
"store_details_csv_schema_file_path"])

#DataTransformConfig
DataTransformationConfig=namedtuple("DataTransformationConfig",
["transformed_train_dir","transformed_validate_dir","transformed_test_dir","preprocessed_object_folder_path",
"preprocessed_object_file_path"])

#ModelTrainerConfig
ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path_cluster_folder","main_cluster_file_path","trained_model_file_path_cluster0","trained_model_file_path_cluster1",
"trained_model_file_path_cluster2","base_accuracy"])



#datapipelineconfig
TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])