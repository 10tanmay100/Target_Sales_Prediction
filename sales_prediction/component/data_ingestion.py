
from sales_prediction.entity.config_entity import DataIngestionConfig
import sys,os
from sales_prediction.exception import sales_project_exception
from sales_prediction.constant import *
from sales_prediction.logger import logging
from sales_prediction.entity.artifact_entity import DataIngestionArtifact
import numpy as np
import pandas as pd
from sales_prediction.util.util import *
class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
            os.makedirs(self.data_ingestion_config.raw_data_dir,exist_ok=True)
        except Exception as e:
            raise sales_project_exception(e,sys)
    def get_data_from_source(self):
        try:
            df1=pd.read_csv("https://raw.githubusercontent.com/10tanmay100/data_for_target_Sales/main/Business_Data.csv")
            df2=pd.read_csv("https://raw.githubusercontent.com/10tanmay100/data_for_target_Sales/main/Sales_History.csv")
            df3=pd.read_csv("https://raw.githubusercontent.com/10tanmay100/data_for_target_Sales/main/Store_Details.csv",encoding='latin1')
            df1.to_csv(os.path.join(self.data_ingestion_config.raw_data_dir,"Business_Data.csv"),index=False)
            df2.to_csv(os.path.join(self.data_ingestion_config.raw_data_dir,"Sales_History.csv"),index=False)
            df3.to_csv(os.path.join(self.data_ingestion_config.raw_data_dir,"Store_Details.csv"),index=False)
        except Exception as e:
            raise sales_project_exception(e,sys) from e


    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            self.get_data_from_source()
            return DataIngestionArtifact(raw_data_file1_path=os.path.join(self.data_ingestion_config.raw_data_dir,"Business_Data.csv"),
            raw_data_file2_path=os.path.join(self.data_ingestion_config.raw_data_dir,"Sales_History.csv"),
            raw_data_file3_path=os.path.join(self.data_ingestion_config.raw_data_dir,"Store_Details.csv"),is_ingested="yes",message="Data ingested successfully")
        except Exception as e:
            raise sales_project_exception(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")



