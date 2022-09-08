import logging
from sales_prediction.exception import sales_project_exception
import yaml
import os,sys
import pandas as pd


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