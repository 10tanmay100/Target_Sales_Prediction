from collections import namedtuple

DataIngestionConfig=namedtuple("DataIngestionConfig",
["raw_data_dir"])

#datapipelineconfig
TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])