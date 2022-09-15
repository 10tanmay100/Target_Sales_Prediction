from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
["raw_data_file1_path","raw_data_file2_path","raw_data_file3_path","is_ingested","message"])


DataValidationArtifact = namedtuple("DataValidationArtifact",
["business_csv_schema_file_path","sales_history_csv_schema_file_path","store_details_csv_schema_file_path","is_validated","message"])



DataTransformationArtifact = namedtuple("DataTransformationArtifact",
["is_transformed", "message", "transformed_train_file_path","transformed_validate_file_path","transformed_test_file_path","preprocessed_object_file_path"])


ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", "message","main_cluster_file_path","trained_model_file_path_cluster0","trained_model_file_path_cluster1","trained_model_file_path_cluster2"])