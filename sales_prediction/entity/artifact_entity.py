from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
["raw_data_file1_path","raw_data_file2_path","raw_data_file3_path","is_ingested","message"])


DataValidationArtifact = namedtuple("DataValidationArtifact",
["business_csv_schema_file_path","sales_history_csv_schema_file_path","store_details_csv_schema_file_path","is_validated","message"])