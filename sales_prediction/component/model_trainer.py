from concurrent.futures.thread import _worker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm # Linear Regression from STATSMODEL
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sales_prediction.exception import sales_project_exception
import sys
from sales_prediction.logger import logging
from typing import List
import shutil
import pickle
from sales_prediction.constant import *
from sales_prediction.util.util import *
from sales_prediction.entity.config_entity import *
from sales_prediction.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise sales_project_exception(e, sys) from e
    
    def start_training_model(self):
        try:
            os.makedirs(self.model_trainer_config.trained_model_file_path_cluster_folder,exist_ok=True)
            #working on training data
            train_df=pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
            logging.info(f"Reading the file {train_df}")
            #applying clustering kmeans here with cluster 8
            kmeans=KMeans(n_clusters=3,init="k-means++",random_state=0)
            logging.info("Kmeans will be used for clustering purpose")
            x=train_df.drop("Total_Sales",axis=1)
            labels=kmeans.fit_predict(x)
            #adding label series in cluster column
            cluster_df=pd.concat([x,train_df["Total_Sales"],pd.DataFrame(labels,columns=["cluster"])],axis=1)
            logging.info("adding label series in cluster column")
            #creating different dataframes for each cluster
            df1=cluster_df[cluster_df["cluster"]==0]
            df2=cluster_df[cluster_df["cluster"]==1]
            df3=cluster_df[cluster_df["cluster"]==2]
        

            with open(self.model_trainer_config.main_cluster_file_path,'wb') as f:
                pickle_file = pickle.dump(kmeans,f)
            shutil.copy(self.model_trainer_config.main_cluster_file_path,ROOT_DIR)
            #cluster1 work
            logging.info("Cluster 1 work has been started!!!")
            df1.drop("cluster",axis=1,inplace=True)
            
            X=df1.drop("Total_Sales",axis=1)
            y=df1["Total_Sales"]

            #read validate csv
            valid=pd.read_csv(self.data_transformation_artifact.transformed_validate_file_path)
            #dropped total sales column
            x1=valid.drop("Total_Sales",axis=1)
            #doing cluster for validating the data
            labels1=kmeans.predict(x1)
            cluster_df_valid=pd.concat([x1,valid["Total_Sales"],pd.DataFrame(labels1,columns=["cluster"])],axis=1)
            df1_valid=cluster_df_valid[cluster_df_valid["cluster"]==0]
            df1_valid.drop("cluster",axis=1,inplace=True)
            X1=df1_valid.drop("Total_Sales",axis=1)
            y1=df1_valid["Total_Sales"]

            model_cluster_1=do_train_cluster_1(X,y,X1,y1)
            with open(self.model_trainer_config.trained_model_file_path_cluster0,'wb') as f:
                pickle_file = pickle.dump(model_cluster_1,f)
            shutil.copy(self.model_trainer_config.trained_model_file_path_cluster0,ROOT_DIR)


            #cluster2 work
            df2.drop("cluster",axis=1,inplace=True)
            XX=df2.drop("Total_Sales",axis=1)
            yy=df2["Total_Sales"]
            df2_valid=cluster_df_valid[cluster_df_valid["cluster"]==1]
            df2_valid.drop("cluster",axis=1,inplace=True)
            X2=df2_valid.drop("Total_Sales",axis=1)
            y2=df2_valid["Total_Sales"]
            model_cluster_2=do_train_cluster_2(XX,yy,X2,y2)

            with open(self.model_trainer_config.trained_model_file_path_cluster1,'wb') as f:
                pickle_file = pickle.dump(model_cluster_2,f)
            shutil.copy(self.model_trainer_config.trained_model_file_path_cluster1,ROOT_DIR)

            #cluster3 work
            df3.drop("cluster",axis=1,inplace=True)
            XXX=df3.drop("Total_Sales",axis=1)
            yyy=df3["Total_Sales"]
            df3_valid=cluster_df_valid[cluster_df_valid["cluster"]==2]
            df3_valid.drop("cluster",axis=1,inplace=True)
            X3=df3_valid.drop("Total_Sales",axis=1)
            y3=df3_valid["Total_Sales"]

            model_cluster_3=do_train_cluster_3(XXX,yyy,X3,y3)
            with open(self.model_trainer_config.trained_model_file_path_cluster2,'wb') as f:
                pickle_file = pickle.dump(model_cluster_3,f)
            shutil.copy(self.model_trainer_config.trained_model_file_path_cluster2,ROOT_DIR)


            return ModelTrainerArtifact(is_trained=True,message="Training has been completed!!",main_cluster_file_path=self.model_trainer_config.main_cluster_file_path,trained_model_file_path_cluster0=self.model_trainer_config.trained_model_file_path_cluster0,trained_model_file_path_cluster1=self.model_trainer_config.trained_model_file_path_cluster1,trained_model_file_path_cluster2=self.model_trainer_config.trained_model_file_path_cluster2
            )

        except Exception as e:
           raise sales_project_exception(e,sys) from e
    
    def initiate_model_training(self):
        self.start_training_model()
