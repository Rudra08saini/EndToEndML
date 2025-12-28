import os
import sys
from src.exception import CustomException
from src.logger import logging
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models , param):
    try:
        report = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, params, cv=skf, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_test_pred = model.predict(X_test)
            
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        logging.error("Error occurred during model evaluation")
        raise CustomException(e, sys)  

def load_object(filepath):
    
    try:
        with open(filepath , "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e , sys) 
           