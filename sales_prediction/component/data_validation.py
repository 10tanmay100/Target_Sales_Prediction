from sales_prediction.logger import logging
from sales_prediction.exception import sales_project_exception
from sales_prediction.entity.config_entity import DataValidationConfig
from sales_prediction.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import os,sys
import pandas as pd
from sales_prediction.constant import *
# from evidently.model_profile import Profile
# from evidently.model_profile.sections import DataDriftProfileSection
# from evidently.dashboard import Dashboard
# from evidently.dashboard.tabs import DataDriftTab
from sales_prediction.util.util import *
import json

class DataValidation:
    

    def __init__(self, data_validation_config:DataValidationConfig,
        data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*30}Data Validation log started.{'<<'*30} \n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.timestamp=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        except Exception as e:
            raise sales_project_exception(e,sys) from e



    def is_raw_file_exists(self)->bool:
        try:
            logging.info("Checking if raw files are available")
            is_business_csv_file_exist = False
            is_sales_history_csv_file_exist = False
            is_store_details_csv_file_exist= False


            business_csv_path = self.data_ingestion_artifact.raw_data_file1_path
            sales_history_csv_path = self.data_ingestion_artifact.raw_data_file2_path
            store_details_csv_path=self.data_ingestion_artifact.raw_data_file3_path


            is_business_csv_file_exist = os.path.exists(business_csv_path)
            is_sales_history_csv_file_exist = os.path.exists(sales_history_csv_path)
            is_store_details_csv_file_exist=os.path.exists(store_details_csv_path)

            is_available =  is_business_csv_file_exist and is_sales_history_csv_file_exist and is_store_details_csv_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")
            
            if not is_available:
                business_csv_file = self.data_ingestion_artifact.raw_data_file1_path
                sales_history_csv_file = self.data_ingestion_artifact.raw_data_file2_path
                store_details_csv_file = self.data_ingestion_artifact.raw_data_file3_path
                message=f"business_csv_file : {business_csv_file} or sales_history_csv_file : {sales_history_csv_file} or store_details_csv_file : {store_details_csv_file}" \
                    "is not present"
                raise Exception(message)
            os.makedirs(self.data_validation_config.artifact_path,exist_ok=True)
            business_data=pd.read_csv(business_csv_path)
            business_data.to_csv(os.path.join(self.data_validation_config.artifact_path,"validated_business_file.csv"))

            sales_history_data=pd.read_csv(sales_history_csv_path)
            sales_history_data.to_csv(os.path.join(self.data_validation_config.artifact_path,"validated_sales_history_file.csv"))

            store_details_data=pd.read_csv(store_details_csv_path)
            store_details_data.to_csv(os.path.join(self.data_validation_config.artifact_path,"validated_store_details.csv"))
            return is_available
        except Exception as e:
            raise sales_project_exception(e,sys) from e

    
    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False

            business_file_name=os.listdir(os.path.join(DATA_INGESTION_LATEST_DIR,os.listdir(DATA_INGESTION_LATEST_DIR)[len(os.listdir(DATA_INGESTION_LATEST_DIR))-1],"raw_data"))[0]

            sales_history_name=os.listdir(os.path.join(DATA_INGESTION_LATEST_DIR,os.listdir(DATA_INGESTION_LATEST_DIR)[len(os.listdir(DATA_INGESTION_LATEST_DIR))-1],"raw_data"))[1]

            store_details_name=os.listdir(os.path.join(DATA_INGESTION_LATEST_DIR,os.listdir(DATA_INGESTION_LATEST_DIR)[len(os.listdir(DATA_INGESTION_LATEST_DIR))-1],"raw_data"))[2]

            #validation check for raw files
            business_csv_df=pd.read_csv(self.data_ingestion_artifact.raw_data_file1_path)
            l=[]
            business_df_val=False
            for cols in business_csv_df.columns:
                yaml=read_yaml_file(BUSINESS_CSV_SCHEMA_FILE_PATH)
                if cols in yaml.keys():
                    if str(business_csv_df[cols].dtype)==yaml[cols]:
                        l.append(True)
            if len(l)==len(yaml.keys()):
                business_df_val=True
            else:
                Exception("On line 87 in data validation")


            sales_history_df=pd.read_csv(self.data_ingestion_artifact.raw_data_file2_path)
            l1=[]
            sales_history_val=False
            for cols in sales_history_df.columns:
                yaml=read_yaml_file(SALES_HISTORY_SCHEMA_FILE_PATH)
                if cols in yaml.keys():
                    if str(sales_history_df[cols].dtype)==yaml[cols]:
                        l1.append(True)
            if len(l1)==len(yaml.keys()):
                sales_history_val=True
            else:
                Exception("On line 101 in data validation")


            store_details_df=pd.read_csv(self.data_ingestion_artifact.raw_data_file3_path)
            l2=[]
            store_details_val=False
            for cols in store_details_df.columns:
                yaml=read_yaml_file(STORE_DETAILS_SCHEMA_FILE_PATH)
                if cols in yaml.keys():
                    if str(store_details_df[cols].dtype)==yaml[cols]:
                        l2.append(True)
            if len(l2)==len(yaml.keys()):
                store_details_val=True
            else:
                Exception("On line 115 in data validation")

            if (business_df_val==True) & (sales_history_val==True) & (store_details_val==True):
                validation_status=True
            return validation_status
        except Exception as e:
            raise sales_project_exception(e,sys) from e

    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            self.is_raw_file_exists()
            validated=self.validate_dataset_schema()

            data_validation_artifact = DataValidationArtifact(
                business_csv_schema_file_path=self.data_validation_config.business_csv_schema_file_path,
                sales_history_csv_schema_file_path=self.data_validation_config.sales_history_csv_schema_file_path,
                store_details_csv_schema_file_path=self.data_validation_config.store_details_csv_schema_file_path,
                is_validated=validated,
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise sales_project_exception(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")