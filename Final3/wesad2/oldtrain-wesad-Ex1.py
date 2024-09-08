# Import the model we are using - Generic
#from sklearn.ensemble import RandomForestRegressor
# Import the model we are using
#from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings
import pickle
import time
from xgboost import XGBRegressor
from sklearn.metrics import f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')

# Reload the dataset with the correct delimiter
dataset = pd.read_csv('/Users/samson/Documents/Final3/wesad2/combined_data.csv')

dataset.fillna(0, inplace=True)

# label_mapping = {"baseline": 0, "meditation": 0, "amusement": 0, "stress": 1}
# dataset['condition'] = dataset['condition'].replace(label_mapping)

# orignial trained features
subCol = ['ECG','Temp']

y = dataset['condition'].copy()
X = dataset[subCol]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Define classifiers and their respective parameter grids for Randomized Search
classifiers = {

    'LR': (LogisticRegression(), {
        'C': np.logspace(-4, 4, 20),  # Regularization parameter
        'penalty': ['l1', 'l2'],  # Penalty ('l1' for Lasso, 'l2' for Ridge)
        'solver': ['liblinear', 'saga']  # Optimization solver
    }),

    'LDA': (LinearDiscriminantAnalysis(), {
        'solver': ['lsqr', 'eigen'],  # Solvers compatible with shrinkage
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]  # Shrinkage parameter (if used)
    }),
    
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }),

    'CART': (DecisionTreeClassifier(), {
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]          
    }),

    'RF': (RandomForestClassifier(), {
        'n_estimators': [100, 300, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),

    'GB': (GradientBoostingClassifier(), {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),

    # 'SVM': (SVC(), {
    #     'C': [0.1, 1, 10],
    #     'gamma': ['scale', 'auto'],
    #     'kernel': ['linear', 'rbf']
    # }),

    'ABC': (AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }),
    'CB': (CatBoostClassifier(), {
        'iterations': [100, 300, 500],
        'learning_rate': [0.01, 0.1, 0.3],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }),

    'XT': (ExtraTreesClassifier(), {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15, None]
    })

}

# Dictionary to store F1 scores for each classifier
f1_scores = {}

# Dictionary to store performance metrics for each classifier
performance_metrics = {}

# Record the start time
start_time = time.time()

# Perform RandomizedSearchCV for each classifier
for clf_name, (clf, param_grid) in classifiers.items():
    print(f"Running Randomized Search for {clf_name}...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    clf_random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=10, cv=kf, scoring='f1_macro', random_state=42, n_jobs=-1)
    clf_random_search.fit(X_train, y_train)  # Fit on training data
    
    # Get the best model
    best_model = clf_random_search.best_estimator_
    
    # Predict on test data
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics on test data
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    
    # Save the metrics
    performance_metrics[clf_name] = {
        'F1 Score': f1,
        'Recall': recall,
        'Precision': precision
    }

    # Save the results for each classifier (added recall and precision)
    clf_results = {
        'cv_results': clf_random_search.cv_results_,
        'best_estimator': clf_random_search.best_estimator_,
        'test_score': clf_random_search.best_score_,
        'best_hyperparameters': clf_random_search.best_params_,
        'test_f1_score': f1,  # Store F1 score on test data
        'test_recall_score': recall,  # Store Recall score
        'test_precision_score': precision  # Store Precision score
    }
    
    # Existing function to save results object
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    save_obj(clf_results, f'/Users/samson/Documents/Final3/wesad2/result1/{clf_name}_F2train_wesad_RS-10kf')

# Print the performance metrics for each classifier
for clf_name, metrics in performance_metrics.items():
    print(f"{clf_name}: F1 Score = {metrics['F1 Score']:.4f}, Recall = {metrics['Recall']:.4f}, Precision = {metrics['Precision']:.4f}")

end_time = time.time()
training_duration = end_time - start_time
print(f"Training duration: {training_duration:.4f} seconds")

