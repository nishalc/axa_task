from pathlib import Path
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import category_encoders as ce

data = pd.read_csv('depression_data.csv')
data.columns = data.columns.str.replace(' ', '_').str.lower()
data["mental_illness"] = np.where(data["history_of_mental_illness"] == "Yes", 1, 0)

X = data[['age', 'marital_status', 'education_level',
       'number_of_children', 'smoking_status', 'physical_activity_level',
       'employment_status', 'income', 'alcohol_consumption', 'dietary_habits',
       'sleep_patterns', 'history_of_substance_abuse', 'family_history_of_depression',
       'chronic_medical_conditions']] 
y = data["mental_illness"]  

# encoding ordinal/categorical variables for use in random forests
mapping_dict = [
    {"col": "marital_status", "mapping": {
        "Single" : 0,
        "Married" : 1,
        "Widowed" : 2,
        "Divorced" : 3
    }},
    {"col": "education_level", "mapping": {
        '''Bachelor's Degree''': 2,
        'High School': 0,
        'Associate Degree': 1,  
        '''Master\'s Degree''': 3,   
        'PhD': 4 
    }},
    {"col": "smoking_status", "mapping": {
        'Non-smoker': 0,   
        'Former': 1,     
        'Current': 2
    }},
    {"col": "physical_activity_level", "mapping": {
        'Sedentary': 0,   
        'Moderate': 1,     
        'Active': 2
    }},
    {"col": "employment_status", "mapping": {
        'Unemployed': 0,   
        'Employed': 1,     
        'Active': 2
    }},
    {"col": "alcohol_consumption", "mapping": {
        'Low': 0,   
        'Moderate': 1,     
        'High': 2
    }},
    {"col": "dietary_habits", "mapping": {
        'Unhealthy': 2,   
        'Moderate': 1,   
        'Healthy': 0
    }},
    {"col": "sleep_patterns", "mapping": {
        'Poor': 0,   
        'Fair': 1,   
        'Good': 2
    }},
    {"col": "history_of_substance_abuse", "mapping": {
        "Yes" : 1,
        "No" : 0
    }},
    {"col": "family_history_of_depression", "mapping": {
        "Yes" : 1,
        "No" : 0
    }},
    {"col": "chronic_medical_conditions", "mapping": {
        "Yes" : 1,
        "No" : 0
    }}
]
encoder = ce.OrdinalEncoder(mapping=mapping_dict)

X_encoded = encoder.fit_transform(X)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_encoded, y, test_size=0.2, random_state=42) # train test split

rf = RandomForestClassifier(criterion="gini", random_state=42)

# training the model with a simple grid search to do a basic optimisation of parameters
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [100, 500],
    'max_features': [0.75]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(X_train_rf, y_train_rf)
y_test_pred_prob = grid_search.best_estimator_.predict_proba(X_test_rf)[:, 1]
y_test_pred = y_test_pred_prob > 0.3
print(f"Best parameters: {grid_search.best_params_}")
print(f'Test Accuracy: {accuracy_score(y_test_rf, y_test_pred):.2f}')
print(f'Test Classification Report:\n{classification_report(y_test_rf, y_test_pred)}')
print(f'Test confusion matrix:\n{confusion_matrix(y_test_rf, y_test_pred)}')
print(f'Test roc_auc_score: {roc_auc_score(y_test_rf, y_test_pred_prob):3f}')