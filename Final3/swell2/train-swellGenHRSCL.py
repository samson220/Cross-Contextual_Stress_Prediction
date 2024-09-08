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
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

# Read in data as pandas dataframe and display first 5 rows
dataset = pd.read_csv('/Users/samson/Documents/Final3/swell2/merged_hrv_stress_labels.csv')
dataset.fillna(0, inplace=True)

label_mapping = {"R": 0, "N": 0, "T": 1, "I": 1}
dataset['Condition'] = dataset['Condition'].replace(label_mapping)

#subCol = ['pNN25','MEAN_RR','HR','MEDIAN_RR','LF_PCT','HF']
subCol = ['HR', 'SCR']

y = dataset['Condition'].copy()
X = dataset[subCol]

def preprocess_inputs(df):
    df = df.copy()
    
    df['Condition'] = df['Condition'].replace(label_mapping)
    
    y = df['Condition'].copy()
    X = df[subCol]
    #X = df.drop('condition', axis=1).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(dataset)


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
    
    # Calculate F1 score on test data
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Save the F1 score
    f1_scores[clf_name] = f1

    # Save the results for each classifier
    clf_results = {
        'cv_results': clf_random_search.cv_results_,
        'best_estimator': clf_random_search.best_estimator_,
        'test_score': clf_random_search.best_score_,
        'best_hyperparameters': clf_random_search.best_params_,
        'test_f1_score': f1  # Store F1 score on test data
    }

    # Save the results for each classifier
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    save_obj(clf_results, f'/Users/samson/Documents/Final3/swell2/results/{clf_name}_F2train_swell2_RS-10kf')

# Record the end time
end_time = time.time()

# Calculate the training duration
training_duration = end_time - start_time

# Print the F1 scores for each classifier
for clf_name, f1_score in f1_scores.items():
    print(f"{clf_name}: F1 Score = {f1_score:.4f}")

print(f"Training duration: {training_duration:.4f} seconds")
