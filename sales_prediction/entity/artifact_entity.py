from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
["raw_data_file1_path","raw_data_file2_path","raw_data_file3_path","is_ingested","message"])