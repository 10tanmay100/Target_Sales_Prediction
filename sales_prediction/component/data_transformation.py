
import pickle
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

## combine processing technqiues
from sklearn.compose import ColumnTransformer
from sales_prediction.exception import sales_project_exception
from sales_prediction.logger import logging
from sales_prediction.entity.config_entity import DataTransformationConfig 
from sales_prediction.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
from sales_prediction.constant import *
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sales_prediction.constant import *
from sales_prediction.util.util import *



class DataTransformation:
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise sales_project_exception(e,sys) from e
    def do_transformation_(self):
        try:
            #training directory
            os.makedirs(self.data_transformation_config.transformed_train_dir,exist_ok=True)

            logging.info(f"Train directory created-->{self.data_transformation_config.transformed_train_dir}")

            #validated directory
            os.makedirs(self.data_transformation_config.transformed_validate_dir,exist_ok=True)

            logging.info(f"validate directory created-->{self.data_transformation_config.transformed_validate_dir}")
            
            #testing directory
            os.makedirs(self.data_transformation_config.transformed_test_dir,exist_ok=True)

            logging.info(f"Test directory created-->{self.data_transformation_config.transformed_test_dir}")

            os.makedirs(self.data_transformation_config.preprocessed_object_folder_path,exist_ok=True)

            logging.info(f"Preprocessing folder path created {self.data_transformation_config.preprocessed_object_folder_path}")


            #work on store details data
            store_details=pd.read_csv(self.data_ingestion_artifact.raw_data_file3_path)
            logging.info(f"Reading csv file from {self.data_ingestion_artifact.raw_data_file3_path}")

            sales_history=pd.read_csv(self.data_ingestion_artifact.raw_data_file2_path)
            logging.info(f"Reading csv file from {self.data_ingestion_artifact.raw_data_file2_path}")

            #converting date column object to date format
            sales_history.Date=pd.to_datetime(sales_history.Date)
            logging.info("converting date column object to date format in sales history column")

            #Read the business data
            business_data=pd.read_csv(self.data_ingestion_artifact.raw_data_file1_path)
            logging.info("Reading csv file from {self.data_ingestion_artifact.raw_data_file1_path}")


            sales_history2=pd.DataFrame(sales_history.groupby(["Store","Date"])["Total_Sales"].sum())
            logging.info("Group by done on sales history data!!")

            sales_history2=sales_history2.reset_index(level=0)
            sales_history2=sales_history2.reset_index(level=0)



            #handiling the data column in business data
            business_data["Date"]=pd.to_datetime(business_data["Date"])
            logging.info("Converting date column object type to date type")
            #taking the column which will not have nulls in the column named as CPI
            business_data=business_data[business_data["CPI"].notnull()]
            logging.info("Taking the dataframe file which column will not have null in the column CPI!!")

            ## merging business data with sales history file
            business_data1=business_data.merge(sales_history2,how="left",on=["Store","Date"])
            logging.info("merging business data with sales history group data")

            #taking test data for analysis
            test_data=business_data1[business_data1.Total_Sales.isnull()]
            test_data=test_data.merge(store_details,on="Store",how="left")
            test_data=test_data.drop(["Address","Area_Code","Location"],axis=1)
            logging.info("Test data ready for future prediction!!!")


            #choosing the data for training and validation splitting
            business_data_final=business_data1[business_data1.Total_Sales.notnull()]
            logging.info("Final Data got collected")


            #MARKDOWN Missing replaced with zero
            business_data_final=business_data_final.fillna(0)
            logging.info("Filling markdowns wiith zero")

            #changing the total sales amount based on markdowns
            business_data_final["Total_Sales"]=business_data_final["Total_Sales"]-(business_data_final["MarkDown1"]+business_data_final["MarkDown2"]+business_data_final["MarkDown3"])
            logging.info("Deducting total sales with markdowns")

            #merge with store details file
            final_data=business_data_final.merge(store_details,on="Store",how="left")
            logging.info("Merging final data with final file")
            
            #encode Holiday
            final_data["Holiday"]=final_data["Holiday"].apply(marker)
            logging.info("Encode holiday column")
            final_data=final_data.drop(["Address","Area_Code","Location"],axis=1)


            #creating pipeline
            numeric_processor=Pipeline(steps=[("imputation_constant",SimpleImputer(missing_values=np.nan,fill_value=0)),("scaler",StandardScaler())])
            
            categorical_processor=Pipeline(steps=[("onehot",OneHotEncoder(drop="first"))])


            final_data=final_data.drop(["Store","Date"],axis=1)

            preprocessor=ColumnTransformer([("categorical",categorical_processor,["Type"]),("numerical",numeric_processor,["Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment_Rate","Holiday","Size"])])

            #splitting the data in train and validation set
            df_train,df_valid=train_test_split(final_data,test_size=0.2,random_state=0)
            logging.info("Splitting the data in train and validate dataset")

            #divide X and y the train data
            X=df_train.drop("Total_Sales",axis=1)
            y=df_train["Total_Sales"]

            #transform the train data
            file=pd.DataFrame(preprocessor.fit_transform(X),columns=["Food","Religion","Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment_Rate","Holiday","Size"])
            logging.info("Transform the train data")
            #concatenate with target sales
            answer=pd.concat([file,y],axis=1)
            logging.info("Concatenation completed!!!!")
            #converted to csv
            answer.to_csv(os.path.join(self.data_transformation_config.transformed_train_dir,"train.csv"),index=False)
            logging.info(f"Train succesfully stored in {self.data_transformation_config.transformed_train_dir} directory")

            #divide the data for validation
            X1=df_valid.drop("Total_Sales",axis=1)
            y1=df_valid["Total_Sales"]
            logging.info("Validation data divided in X and y")

            # validation data transformed succesfully
            file1=pd.DataFrame(preprocessor.fit_transform(X1),columns=["Food","Religion","Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment_Rate","Holiday","Size"])
            logging.info("Data Validated successfully!!!")

            #concatenation with target column with validate csv
            answer1=pd.concat([file1,y1],axis=1)
            logging.info("Concatenation done on valid data")


            #convert the dataframe into csv file
            answer1.to_csv(os.path.join(self.data_transformation_config.transformed_validate_dir,"validated.csv"),index=False)
            logging.info(f"validate data stored in {answer1}")


            test_data=pd.DataFrame(preprocessor.transform(test_data),columns=["Food","Religion","Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","CPI","Unemployment_Rate","Holiday","Size"])
            test_data.to_csv(os.path.join(self.data_transformation_config.transformed_test_dir,"test.csv"),index=False)

            path_pkl=self.data_transformation_config.preprocessed_object_file_path
            with open(path_pkl,"wb") as f:
                pickle.dump(preprocessor,f)
            shutil.copy(path_pkl,ROOT_DIR)

            return DataTransformationArtifact(is_transformed=True,message="Data Transformed",transformed_train_file_path=os.path.join(self.data_transformation_config.transformed_train_dir,"train.csv"),transformed_validate_file_path=os.path.join(self.data_transformation_config.transformed_validate_dir,"validated.csv"),transformed_test_file_path=os.path.join(self.data_transformation_config.transformed_test_dir,"test.csv"),preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path)

        except Exception as e:
            raise sales_project_exception(e,sys) from e


    def initiate_data_transformation(self):
        try:
            data_transformation_artifact=self.do_transformation_()
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise sales_project_exception(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")


