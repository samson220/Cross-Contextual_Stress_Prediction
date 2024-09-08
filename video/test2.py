import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv('/Users/samson/Documents/Final2/video/participant_till4.csv')

# Convert timestamp columns to numerical values
time_columns = ['Time_Biotrace', 'Time_Videorating', 'Time_Light', 'Time_Accel', 'Time_GPS']
for col in time_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce').view('int64') // 10**9

# Define a custom function to categorize 'Rating_Videorating'
def categorize_rating(rating):
    if rating <= 200:
        return 0
    elif rating <= 300:
        return 1
    elif rating <= 400:
        return 2
    else:
        return 3

# Apply the function to create a new 'stress' column
data['stress'] = data['Rating_Videorating'].apply(categorize_rating)

# Define features and target
X = data.drop(['Rating_Videorating', 'stress'], axis=1)
y = data['stress']

# Create a standard scaler transformer for the numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features)
], remainder='passthrough')

# Create a pipeline with preprocessing and a classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter space for RandomizedSearchCV
param_distributions = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=10
)

# Fit the model
random_search.fit(X_train, y_train)

# The best hyperparameters from RandomizedSearchCV
best_params = random_search.best_params_
print(best_params)
