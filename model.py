# imports
import pandas as pd
import numpy as np

#preprocessing
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# modeling imports
from sklearn.ensemble import (GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# evalutaion metrics
from sklearn.metrics import (accuracy_score, 
                             auc, 
                             precision_score,
                             recall_score,
                             f1_score, 
                             roc_auc_score,
                             confusion_matrix)

# data
from wrangle import wrangle_data


# encoding function
def encode_features(X_list):
    '''
    Arguments: [X_train, X_validate, X_test]
    Actions:
        1. Encodes variables
        2. Creates new datasets with encoded variables  
    Returns: [X_train_encoded, X_validate_encoded, X_test_encoded] 
    Modules:
        1. import pandas as pd
        2. from sklearn.preprocessing import LabelEncoder
    '''
    # set ordinal variables
    ordinal = ['age_bins',  'glucose_bins', 'bmi_bins', 'hypertension', 'heart_disease']
    
    # set nominal variables
    nominal = ['ever_married', 'work_type']
    
    # initialize encoder
    le = LabelEncoder()
    
    # for each dataset
    for X in X_list:
        
        # for each ordinal variable in each dataset
        for col in ordinal:
            
            # fit and transform each and replace the values in the original
            X[col] = le.fit_transform(X[col])
    
    # initialize list
    converted = []
    
    # for each data set
    for X in X_list:
        
        # create temporrary dataset with pre-encoded variables
        temp = X.drop(nominal, axis=1)

        # get the dummy variables for each nominal variable
        dummies = pd.get_dummies(X[nominal], drop_first=True)

        # add new datasets with all encoded variables to the list
        converted.append(pd.concat([temp, dummies], axis = 1))
       
    # exit function and return the list of encoded datasets
    return converted
    


# preprocess
def preprocess():
    '''
    Actions:
        1. Gets data
        2. Creates X, y datasets for train, validate, and test
        3. Encodes all X datasets
        4. Oversamples using X train and y train
    Returns: X_train, y_train, X_resample, y_resample, X_validate, y_validate, X_test, y_test
    Modules:
        1. import pandas as pd
        2. from wrangle import wrangle_data
        3. from model import encode_features
        4. from imblearn.over_sampling import SMOTE
    '''
    # get data
    train, validate, test = wrangle_data()
    
    # set target
    target = 'stroke'
    
    # set features of interest
    features = ['hypertension', 'heart_disease', 'ever_married', 'work_type', 'age_bins', 'glucose_bins', 'bmi_bins']
    
    # create train X, y
    X_train = train[features]
    y_train = train[target]
    
    # create validate X, y
    X_validate = validate[features]
    y_validate = validate[target]
    
    # create test X, y
    X_test = test[features]
    y_test = test[target]
    
    # encoding variables
    X_train, X_validate, X_test = encode_features([X_train, X_validate, X_test])
    
    # initialize oversampling 
    smote = SMOTE(random_state=1017)
    # fit and resample using X and y train
    X_resample, y_resample = smote.fit_resample(X_train, y_train.ravel())
    
    # exit function and return all preprocessed datasets
    return X_train, y_train, X_resample, y_resample, X_validate, y_validate, X_test, y_test


#### NEW
#### predictions with resampled data

#### predictions with resampled data

def predictions(x_set,y_set, X_validate, y_validate):
    '''
    Actions: Gets dataframe with evaluation scores for SVC, GradientBoost, and LogisticRegression classifiers
    '''
    
    # initialize lists to hold metrics
    accuracy,precision,recall,f1,conf_mat= [],[],[],[],[]
    
    # set a random state
    random_state = 1017
    
    # set baseline predictions
    y_preds = np.zeros(len(X_validate)).astype(int)

    # adding metrics for baseline
    accuracy.append((round(accuracy_score(y_validate,y_preds),2))*100)
    precision.append((round(precision_score(y_validate,y_preds),2))*100)
    recall.append((round(recall_score(y_validate,y_preds),2))*100)
    f1.append((round(f1_score(y_validate,y_preds),2))*100)
    conf_mat.append(confusion_matrix(y_validate,y_preds))

    
    # intitializing different classifiers
    clf1 = SVC(random_state=random_state, probability=True)
    clf2 = GradientBoostingClassifier(random_state=random_state)
    clf3 = LogisticRegression(random_state = random_state)
    clf4 = LogisticRegression(C=.25, random_state = random_state)
    clf5 = LogisticRegression(C=.5, random_state = random_state)

    # initializing voting classifier with top three classifiers from above
    eclf = VotingClassifier(estimators=[
        ('svc', clf1),('gbc', clf2), ('lr', clf3), ('lr.5', clf4), ('lr.25', clf5)])
    
    
    # initialize classifier list
    classifiers = []
    
    # adding classification models to be used
    classifiers.append(clf1)
    classifiers.append(clf2)
    classifiers.append(clf3)    
    classifiers.append(clf4)
    classifiers.append(clf5)
    classifiers.append(eclf)
    
    # for each classification method in the list
    for clf in classifiers:
        
        # fit classifier
        clf.fit(x_set,y_set)
        
        # assign predictions to variable
        y_preds = clf.predict(X_validate)
        
        # appending the metrics to each repsective metric list
        accuracy.append((round(accuracy_score(y_validate,y_preds),2))*100)
        precision.append((round(precision_score(y_validate,y_preds),2))*100)
        recall.append((round(recall_score(y_validate,y_preds),2))*100)
        f1.append((round(f1_score(y_validate,y_preds),2))*100)
        conf_mat.append(confusion_matrix(y_validate,y_preds))

    # creating a dataframe with the metrics from the list and each algorithm name
    results_df = pd.DataFrame({"Recall Score":recall,
                               "Accuracy Score":accuracy,
                               "Precision Score":precision,
                               "f1 Score":f1,
                               "Confusion Matrix":conf_mat,
                               "Algorithm":["Baseline",
                                            "SVC",
                                            "GradientBoosting",
                                            "LogisticRegression",
                                            "LR C=.25",
                                            "LR C=.5",
                                            "VotingClassifier"]})
                                     
    # sorting algorithm name alphabetically and setting index to the algorithm name 
    results_df = results_df.sort_values(by = 'Algorithm').set_index('Algorithm')
    
    # exit function and return df
    return results_df

#### predictions with resampled data NEW

# def predictions(x_set,y_set, X_validate, y_validate):
#     '''
#     Actions: Gets dataframe with evaluation scores for SVC, GradientBoost, and LogisticRegression classifiers
#     '''
    
#     # initialize lists to hold metrics
#     accuracy,precision,recall,f1,conf_mat= [],[],[],[],[]
    
#     # set a random state
#     random_state = 1017
    
#     # set baseline predictions
#     y_preds = np.zeros(len(X_validate)).astype(int)

#     # adding metrics for baseline
#     accuracy.append((round(accuracy_score(y_validate,y_preds),2))*100)
#     precision.append((round(precision_score(y_validate,y_preds),2))*100)
#     recall.append((round(recall_score(y_validate,y_preds),2))*100)
#     f1.append((round(f1_score(y_validate,y_preds),2))*100)
#     conf_mat.append(confusion_matrix(y_validate,y_preds))

    
#     # intitializing different classifiers
#     clf1 = SVC(random_state=random_state, probability=True)
#     clf2 = GradientBoostingClassifier(random_state=random_state)
#     clf3 = LogisticRegression(random_state = random_state)

#     # initializing voting classifier with top three classifiers from above
#     eclf = VotingClassifier(estimators=[
#         ('svc', clf1), ('gbc', clf2), ('lr', clf3)])
    
    
#     # initialize classifier list
#     classifiers = []
    
#     # adding classification models to be used
#     classifiers.append(clf1)
#     classifiers.append(clf2)
#     classifiers.append(clf3)
#     classifiers.append(eclf)
    
#     # for each classification method in the list
#     for clf in classifiers:
        
#         # fit classifier
#         clf.fit(x_set,y_set)
        
#         # assign predictions to variable
#         y_preds = clf.predict(X_validate)
        
#         # appending the metrics to each repsective metric list
#         accuracy.append((round(accuracy_score(y_validate,y_preds),2))*100)
#         precision.append((round(precision_score(y_validate,y_preds),2))*100)
#         recall.append((round(recall_score(y_validate,y_preds),2))*100)
#         f1.append((round(f1_score(y_validate,y_preds),2))*100)
#         conf_mat.append(confusion_matrix(y_validate,y_preds))

#     # creating a dataframe with the metrics from the list and each algorithm name
#     results_df = pd.DataFrame({"Recall Score":recall,
#                                "Accuracy Score":accuracy,
#                                "Precision Score":precision,
#                                "f1 Score":f1,
#                                "Confusion Matrix":conf_mat,
#                                "Algorithm":["Baseline",
#                                             "SVC",
#                                             "GradientBoosting",
#                                             "LogisticRegression",
#                                             "VotingClassifier"]})
                                     
#     # sorting algorithm name alphabetically and setting index to the algorithm name 
#     results_df = results_df.sort_values(by = 'Algorithm').set_index('Algorithm')
    
#     # exit function and return df
#     return results_df


# voting classifier
def voting_predictions(X_train, y_train, X_validate, y_validate):
    '''
    Actions: Gets dataframe with evaluation scores for VotingClassifier that uses SVC, GradientBoost, and LogisticRegression classifiers as voting parties
    '''
    # setting random state
    random_state = 1017
    
 # intitializing different classifiers
    clf1 = SVC(random_state=random_state, probability=True)
    clf2 = GradientBoostingClassifier(random_state=random_state)
    clf3 = LogisticRegression(random_state = random_state)
    clf4 = LogisticRegression(C=.25, random_state = random_state)
    clf5 = LogisticRegression(C=.5, random_state = random_state)

    # initializing voting classifier with top three classifiers from above
    eclf = VotingClassifier(estimators=[
        ('svc', clf1),('gbc', clf2), ('lr', clf3), ('lr.5', clf4), ('lr.25', clf5)])

    # fitting the model on the resampled train data
    eclf.fit(X_train, y_train)

    # assign predictions to variable
    y_preds = eclf.predict(X_validate)

    # initialize lists to hold metrics
    accuracy,precision,recall,f1,conf_mat= [],[],[],[],[]

    # appending the metrics to each repsective metric list
    accuracy.append((round(accuracy_score(y_validate,y_preds),2))*100)
    precision.append((round(precision_score(y_validate,y_preds),2))*100)
    recall.append((round(recall_score(y_validate,y_preds),2))*100)
    f1.append((round(f1_score(y_validate,y_preds),2))*100)
    conf_mat.append(confusion_matrix(y_validate,y_preds))

    # creating a dataframe with the metrics from the list and each algorithm name
    results_df = pd.DataFrame({"Recall Score":recall,
                               "Accuracy Score":accuracy,
                               "Precision Score":precision,
                               "f1 Score":f1,
                               "Confusion Matrix":conf_mat,
                               "Algorithm":'VotingClassifier_uniform'})

    # sorting algorithm name alphabetically and setting index to the algorithm name 
    return results_df.sort_values(by = 'Algorithm').set_index('Algorithm').T

