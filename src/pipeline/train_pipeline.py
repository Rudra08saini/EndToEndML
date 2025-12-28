import os 
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import save_object , evaluate_models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
    
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params ={
                
                "Random Forest": {
                    'n_estimators':[8,16,32,64,128,256],
                    'max_features':['sqrt','log2',None],
                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                    
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'] ,
                    'max_features':['sqrt','log2']
                },
                "Gradient Boosting": {
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256],
                    'max_features':['sqrt','log2',None],
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile']
                },
                "Linear Regression": {},
                "KNeighbors Regressor": {
                    'n_neighbors':[5,7,9,11],
                    'weights':['uniform','distance']
                },
                "XGB Regressor": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                
                
                "AdaBoost Regressor": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models , param=params)
            
            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with r2 score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)    