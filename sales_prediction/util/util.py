import logging
from sales_prediction.exception import sales_project_exception
import yaml
import os,sys
import pandas as pd
import numpy as np
import seaborn as sns
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


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise sales_project_exception(e,sys) from e

def marker(x):
    if x==True:
        return 1
    return 0

def do_train_cluster_1(X_true,y_true,X_valid,y_valid):
    model_name=[]
    models=[]
    scores=[]
    lr=LinearRegression()
    lr.fit(X_true,y_true)
    train_pred=lr.predict(X_true)
    r2_score(y_true,train_pred)
    prediction=lr.predict(X_valid)
    lr_score=r2_score(y_valid,prediction)
    model_name.append("Linear Regression")
    models.append(lr)
    scores.append(lr_score)

    lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
    lasscv.fit(X_true,y_true)
    alpha = lasscv.alpha_
    lasso_reg = Lasso(alpha)
    lasso_reg.fit(X_true,y_true)
    prediction_lasso=lasso_reg.predict(X_true)
    r2_score(y_true,prediction_lasso)
    prediction_lasso_valid=lasso_reg.predict(X_valid)
    lasso_score=r2_score(y_valid,prediction_lasso_valid)
    model_name.append("Lasso")
    models.append(lasso_reg)
    scores.append(lasso_score)

    alphas = np.random.uniform(low=0, high=10, size=(50,))
    ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
    ridgecv.fit(X_true,y_true)
    prediction_ridge=ridgecv.predict(X_true)
    r2_score(y_true,prediction_ridge)
    prediction_ridge_valid=ridgecv.predict(X_valid)
    ridge_score=r2_score(y_valid,prediction_ridge_valid)
    model_name.append("Ridge")
    models.append(ridgecv)
    scores.append(ridge_score)

    #commented because of taking time to execute this I have put
    # parameters manually after the tuning
    # svr=SVR()
    # params={"kernel":['linear', 'poly', 'rbf', 'sigmoid'],"degree":[1,2,3],"gamma":['scale', 'auto'],"tol":[1e-1,1e-2,1e-3,1e-4,1e-5],"C":[0.5,0.6,0.7,0.8,0.9,1.0]}
    # g=GridSearchCV(estimator=svr,param_grid=params,cv=100)
    # g.fit(X_true,y_true)
    # g.best_params_
    svr=SVR(kernel="linear",gamma="auto",tol=1e-4,C=0.5)
    svr.fit(X_true,y_true)
    prediction_svr=svr.predict(X_true)
    r2_score(y_true,prediction_svr)
    prediction_svr_valid=svr.predict(X_valid)
    svr_score=r2_score(y_valid,prediction_svr_valid)
    model_name.append("svr")
    models.append(svr)
    scores.append(svr_score)

    #commented because of taking time to execute this I have put
    # parameters manually after the tuning
    # dr=DecisionTreeRegressor()
    # params={"criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"splitter":['best', 'random'],"max_depth":[2,3,4,5,6,7,8,9,10],"min_samples_split":[0.5,0.6,0.7,2,3,4,5],"min_samples_leaf":[2,3,4,5,6],"max_features":['auto', 'sqrt', 'log2'],"max_leaf_nodes":[1,2,3,4,5,6,7,8,9,10]}
    # g=GridSearchCV(estimator=dr,param_grid=params,cv=10)
    # g.fit(X_true,y_true)
    # g.best_params_
    dt_model1 = DecisionTreeRegressor(max_depth=6,min_samples_split=20,min_samples_leaf=20,max_leaf_nodes=7,max_features=0.8,random_state=0)
    dt_model1.fit(X_true,y_true)
    dt_model1.score(X_true,y_true)
    dt_score=dt_model1.score(X_valid, y_valid)
    model_name.append("Decision Tree")
    models.append(dt_model1)
    scores.append(dt_score)

    knn=KNeighborsRegressor(n_neighbors=4,algorithm="brute")
    knn.fit(X_true,y_true)
    knn.score(X_true,y_true)
    knn_score=knn.score(X_valid, y_valid)
    model_name.append("knn")
    models.append(knn)
    scores.append(knn_score)
    f=open("cluster1.txt","a")
    f.write(f"selected model is {model_name[scores.index(max(scores))]}")

    #find the max validated accuracy here 
    model_cluster1=models[scores.index(max(scores))]
    return model_cluster1



def do_train_cluster_2(X_true,y_true,X_valid,y_valid):
    model_name=[]
    models=[]
    scores=[]
    lr=LinearRegression()
    lr.fit(X_true,y_true)
    train_pred=lr.predict(X_true)
    r2_score(y_true,train_pred)
    prediction=lr.predict(X_valid)
    lr_score=r2_score(y_valid,prediction)
    model_name.append("Linear Regression")
    models.append(lr)
    scores.append(lr_score)

    lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
    lasscv.fit(X_true,y_true)
    alpha = lasscv.alpha_
    lasso_reg = Lasso(alpha)
    lasso_reg.fit(X_true,y_true)
    prediction_lasso=lasso_reg.predict(X_true)
    r2_score(y_true,prediction_lasso)
    prediction_lasso_valid=lasso_reg.predict(X_valid)
    lasso_score=r2_score(y_valid,prediction_lasso_valid)
    model_name.append("Lasso")
    models.append(lasso_reg)
    scores.append(lasso_score)

    alphas = np.random.uniform(low=0, high=10, size=(50,))
    ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
    ridgecv.fit(X_true,y_true)
    prediction_ridge=ridgecv.predict(X_true)
    r2_score(y_true,prediction_ridge)
    prediction_ridge_valid=ridgecv.predict(X_valid)
    ridge_score=r2_score(y_valid,prediction_ridge_valid)
    model_name.append("Ridge")
    models.append(ridgecv)
    scores.append(ridge_score)

    #commented because of taking time to execute this I have put
    # parameters manually after the tuning
    # svr=SVR()
    # params={"kernel":['linear', 'poly', 'rbf', 'sigmoid'],"degree":[1,2,3],"gamma":['scale', 'auto'],"tol":[1e-1,1e-2,1e-3,1e-4,1e-5],"C":[0.5,0.6,0.7,0.8,0.9,1.0]}
    # g=GridSearchCV(estimator=svr,param_grid=params,cv=100)
    # g.fit(X_true,y_true)
    # g.best_params_
    svr=SVR(kernel="linear",gamma="auto",tol=1e-4,C=0.5)
    svr.fit(X_true,y_true)
    prediction_svr=svr.predict(X_true)
    r2_score(y_true,prediction_svr)
    prediction_svr_valid=svr.predict(X_valid)
    svr_score=r2_score(y_valid,prediction_svr_valid)
    model_name.append("svr")
    models.append(svr)
    scores.append(svr_score)

    #commented because of taking time to execute this I have put
    # parameters manually after the tuning
    # dr=DecisionTreeRegressor()
    # params={"criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"splitter":['best', 'random'],"max_depth":[2,3,4,5,6,7,8,9,10],"min_samples_split":[0.5,0.6,0.7,2,3,4,5],"min_samples_leaf":[2,3,4,5,6],"max_features":['auto', 'sqrt', 'log2'],"max_leaf_nodes":[1,2,3,4,5,6,7,8,9,10]}
    # g=GridSearchCV(estimator=dr,param_grid=params,cv=10)
    # g.fit(X_true,y_true)
    # g.best_params_
    dt_model1 = DecisionTreeRegressor(max_depth=3,max_leaf_nodes=7,max_features=0.9,random_state=0)
    dt_model1.fit(X_true,y_true)
    dt_model1.score(X_true,y_true)
    dt_score=dt_model1.score(X_valid, y_valid)
    model_name.append("Decision Tree")
    models.append(dt_model1)
    scores.append(dt_score)

    knn=KNeighborsRegressor(n_neighbors=4,algorithm="auto")
    knn.fit(X_true,y_true)
    knn.score(X_true,y_true)
    knn_score=knn.score(X_valid, y_valid)
    model_name.append("knn")
    models.append(knn)
    scores.append(knn_score)
    f=open("cluster2.txt","a")
    f.write(f"selected model is {model_name[scores.index(max(scores))]}")

    #find the max validated accuracy here 
    model_cluster2=models[scores.index(max(scores))]
    return model_cluster2

def do_train_cluster_3(X_true,y_true,X_valid,y_valid):
    model_name=[]
    models=[]
    scores=[]
    lr=LinearRegression()
    lr.fit(X_true,y_true)
    train_pred=lr.predict(X_true)
    r2_score(y_true,train_pred)
    prediction=lr.predict(X_valid)
    lr_score=r2_score(y_valid,prediction)
    model_name.append("Linear Regression")
    models.append(lr)
    scores.append(lr_score)

    lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
    lasscv.fit(X_true,y_true)
    alpha = lasscv.alpha_
    lasso_reg = Lasso(alpha)
    lasso_reg.fit(X_true,y_true)
    prediction_lasso=lasso_reg.predict(X_true)
    r2_score(y_true,prediction_lasso)
    prediction_lasso_valid=lasso_reg.predict(X_valid)
    lasso_score=r2_score(y_valid,prediction_lasso_valid)
    model_name.append("Lasso")
    models.append(lasso_reg)
    scores.append(lasso_score)

    alphas = np.random.uniform(low=0, high=10, size=(50,))
    ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
    ridgecv.fit(X_true,y_true)
    prediction_ridge=ridgecv.predict(X_true)
    r2_score(y_true,prediction_ridge)
    prediction_ridge_valid=ridgecv.predict(X_valid)
    ridge_score=r2_score(y_valid,prediction_ridge_valid)
    model_name.append("Ridge")
    models.append(ridgecv)
    scores.append(ridge_score)

    #commented because of taking time to execute this I have put
    # parameters manually after the tuning
    # svr=SVR()
    # params={"kernel":['linear', 'poly', 'rbf', 'sigmoid'],"degree":[1,2,3],"gamma":['scale', 'auto'],"tol":[1e-1,1e-2,1e-3,1e-4,1e-5],"C":[0.5,0.6,0.7,0.8,0.9,1.0]}
    # g=GridSearchCV(estimator=svr,param_grid=params,cv=100)
    # g.fit(X_true,y_true)
    # g.best_params_
    svr=SVR(kernel="linear",gamma="auto",tol=1e-4,C=0.5)
    svr.fit(X_true,y_true)
    prediction_svr=svr.predict(X_true)
    r2_score(y_true,prediction_svr)
    prediction_svr_valid=svr.predict(X_valid)
    svr_score=r2_score(y_valid,prediction_svr_valid)
    model_name.append("svr")
    models.append(svr)
    scores.append(svr_score)

    #commented because of taking time to execute this I have put
    # parameters manually after the tuning
    # dr=DecisionTreeRegressor()
    # params={"criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"splitter":['best', 'random'],"max_depth":[2,3,4,5,6,7,8,9,10],"min_samples_split":[0.5,0.6,0.7,2,3,4,5],"min_samples_leaf":[2,3,4,5,6],"max_features":['auto', 'sqrt', 'log2'],"max_leaf_nodes":[1,2,3,4,5,6,7,8,9,10]}
    # g=GridSearchCV(estimator=dr,param_grid=params,cv=10)
    # g.fit(X_true,y_true)
    # g.best_params_
    dt_model1 = DecisionTreeRegressor(max_depth=4,max_leaf_nodes=7,random_state=0)
    dt_model1.fit(X_true,y_true)
    dt_model1.score(X_true,y_true)
    dt_score=dt_model1.score(X_valid, y_valid)
    model_name.append("Decision Tree")
    models.append(dt_model1)
    scores.append(dt_score)

    knn=KNeighborsRegressor(n_neighbors=5,algorithm="auto")
    knn.fit(X_true,y_true)
    knn.score(X_true,y_true)
    knn_score=knn.score(X_valid, y_valid)
    model_name.append("knn")
    models.append(knn)
    scores.append(knn_score)
    f=open("cluster3.txt","a")
    f.write(f"selected model is {model_name[scores.index(max(scores))]}")

    #find the max validated accuracy here 
    model_cluster3=models[scores.index(max(scores))]
    return model_cluster3
