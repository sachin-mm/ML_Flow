import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    return precision,recall,f1,accuracy

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL

    try:
        data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_df.csv"))
        df_target = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "target.csv"))
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    X_train, X_test, y_train, y_test = train_test_split(data, df_target, random_state = 10, test_size = 0.2)

  
    max_leaf_nodes =int(sys.argv[1])
    with mlflow.start_run():
        decision_tree_classification = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes =max_leaf_nodes, random_state = 10)
        decision_tree1 = decision_tree_classification.fit(X_train, y_train)
        y_pred=decision_tree1.predict(X_test)

        (precision,recall,f1,accuracy) = eval_metrics(y_test,y_pred)


        print("  precision: %s" % precision)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)
        print(" accuracy:%s"% accuracy)

        mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(decision_tree_classification, "model", registered_model_name="DecisionTreeClassifier")
        else:
            mlflow.sklearn.log_model(decision_tree_classification, "model")
