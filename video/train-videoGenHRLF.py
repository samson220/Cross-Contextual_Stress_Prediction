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
warnings.filterwarnings('ignore')

# Reload the dataset with the correct delimiter
data = pd.read_csv('/Users/samson/Documents/Final2/video/concatenated_file.csv')

# Define a custom function to categorize 'Rating_Videorating'
def categorize_rating(rating):
    if rating <= 200:
        return 0
    elif rating <= 300:
        return 0
    elif rating <= 400:
        return 1
    else:
        return 2

# Apply the function to create a new 'stress' column
data['stress'] = data['Rating_Videorating'].apply(categorize_rating)
subCol = ['HR','LF']

# Define features and target
# X = data.drop(['Rating_Videorating', 'stress'], axis=1)
X = data[subCol]
y = data['stress']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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

    'abc': (AdaBoostClassifier(), {
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

# Record the start time
start_time = time.time()

# Perform RandomizedSearchCV for each classifier
for clf_name, (clf, param_grid) in classifiers.items():
    print(f"Running Randomized Search for {clf_name}...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    clf_random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=10, cv=kf, scoring='f1_macro', random_state=42, n_jobs=-1)
    clf_random_search.fit(X, y)
    
    # Store cross-validation results, estimators, and test scores separately for each classifier
    clf_results = {
        'cv_results': clf_random_search.cv_results_,
        'best_estimator': clf_random_search.best_estimator_,
        'best_params': clf_random_search.best_params_,
        'test_score': clf_random_search.best_score_
    }

    # Save the results for each classifier
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    save_obj(clf_results, f'/Users/samson/Documents/Final2/video/resultshrlf/{clf_name}_F2train_swellhrlf_RS-10kf')

# Record the end time
end_time = time.time()

# Calculate the training duration
training_duration = end_time - start_time

print(f"Training duration: {training_duration:.4f} seconds")
