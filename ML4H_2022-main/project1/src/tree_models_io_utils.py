import os
import joblib
import numpy as np
import lightgbm as lgb
from xgboost.sklearn import XGBClassifier
from src.constants import *


#method to set the folder where to read/write the data
def set_folder(dataset_name):
  if dataset_name == "MITBIH": folder_name = FOLDER_NAME_MITBIH
  else: folder_name = FOLDER_NAME_PTBDB
  return folder_name

#method to save the random forrest model compressed
def save_rf(model, dataset):
  folder_name = set_folder(dataset)
  joblib.dump(model, folder_name+'rf.joblib', compress=1)

#method to return the random forrest model
def load_rf(dataset):
  folder_name = set_folder(dataset)
  return joblib.load(folder_name + 'rf.joblib')

def save_xgboost(model, dataset):
  folder_name = set_folder(dataset)
  joblib.dump(model, folder_name+'xg.joblib', compress=1)
  #model.save_model(folder_name+"xg.txt")

def load_xgboost(dataset):
  folder_name = set_folder(dataset)
  return joblib.load(folder_name + 'xg.joblib')

#method to save the lgbm model compressed
def save_lgbm(model, dataset):
  folder_name = set_folder(dataset)
  joblib.dump(model, folder_name+'lgbm.joblib', compress=1)

#method to return the lgbm model
def load_lgbm(dataset):
  folder_name = set_folder(dataset)
  return joblib.load(folder_name + 'lgbm.joblib')
