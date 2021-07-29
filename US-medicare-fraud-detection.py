import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import LogisticRegressionCV
import sklearn.metrics as metrics
from sklearn.metrics import cohen_kappa_score,roc_auc_score,f1_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pd.set_option('display.max_columns', 100)

import_status = 0

if import_status == 0:
    # Provider data for training dataset
    train_data0 = pd.read_csv(r'Data_Train/Medicare_Outpatient_Inpatient_Beneficiary_PartB.csv', dtype=object)
    # Provider data for evaluation dataset
    eval_data0 = pd.read_csv(r'Data_Test/Medicare_Outpatient_Inpatient_Beneficiary_Eval_PartB.csv', dtype=object)
    # Inpatient, Outpatient and Beneficiary data for training dataset
    train_prov0 = pd.read_csv(r'Data_Train/Medicare_Provider_PartB.csv', dtype=object)
    # Inpatient, Outpatient and Beneficiary data for evaluation dataset
    eval_prov0 = pd.read_csv(r'Data_Test/Medicare_Provider_Eval_PartB.csv', dtype=object)
    import_status = 1

train_data = copy.deepcopy(train_data0)
eval_data = copy.deepcopy(eval_data0)
train_prov = copy.deepcopy(train_prov0)
eval_prov = copy.deepcopy(eval_prov0)

shape_status = 1
if shape_status == 0:
    print(train_data.shape)
    print(train_data.isna().sum())
    print("")
    print(eval_data.shape)
    print(eval_data.isna().sum())
    print("")
    print(train_prov.isna().sum())
    print("")
    print(eval_prov.isna().sum())
    shape_status = 1

# Remove columns that are completely NA (clmprocedures 5&6)
train_data.dropna(axis=1, how='all', inplace=True)
eval_data.dropna(axis=1, how='all', inplace=True)

# Replace 2 with 0 for chronic conditions, so 0 = no chronic condition and 1 = yes
train_data = train_data.replace(
    {'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
     'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2,
     'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2,
     'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2}, 0)

eval_data = eval_data.replace(
    {'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
     'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2,
     'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2,
     'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2}, 0)

# Replace renal disease indicator "Y" with 1, 0 = no, 1 = yes
train_data = train_data.replace({'RenalDiseaseIndicator': 'Y'}, 1)
eval_data = eval_data.replace({'RenalDiseaseIndicator': 'Y'}, 1)

# Replace claimtype "inpatient" with 0, "outpatient" with 1
train_data = train_data.replace({'ClaimType': 'Inpatient'}, 0)
eval_data = eval_data.replace({'ClaimType': 'Inpatient'}, 0)

train_data = train_data.replace({'ClaimType': 'Outpatient'}, 1)
eval_data = eval_data.replace({'ClaimType': 'Outpatient'}, 1)

# add age column using dod and dob
train_data['DOB'] = pd.to_datetime(train_data['DOB'], format='%Y-%m-%d')
train_data['DOD'] = pd.to_datetime(train_data['DOD'], format='%Y-%m-%d', errors='ignore')
train_data['Age'] = round(((train_data['DOD'] - train_data['DOB']).dt.days) / 365)

eval_data['DOB'] = pd.to_datetime(eval_data['DOB'], format='%Y-%m-%d')
eval_data['DOD'] = pd.to_datetime(eval_data['DOD'], format='%Y-%m-%d', errors='ignore')
eval_data['Age'] = round(((eval_data['DOD'] - eval_data['DOB']).dt.days) / 365)

# if dod = na, then age = dob and current date of 2009-12-31
curr_date = '2009-12-31'
train_data['Age'] = train_data['Age'].fillna(
    round(((pd.to_datetime(curr_date, format='%Y-%m-%d') - train_data['DOB']).dt.days) / 365))

eval_data['Age'] = eval_data['Age'].fillna(
    round(((pd.to_datetime(curr_date, format='%Y-%m-%d') - eval_data['DOB']).dt.days) / 365))

# add a death flag for each person
train_data.loc[train_data.DOD.notna(), 'DeathFlag'] = 1
train_data.loc[train_data.DOD.isna(), 'DeathFlag'] = 0

eval_data.loc[eval_data.DOD.notna(), 'DeathFlag'] = 1
eval_data.loc[eval_data.DOD.isna(), 'DeathFlag'] = 0

# add claim days column using claim start and claim end, note these dates are equal to admission dates
train_data['ClaimStartDt'] = pd.to_datetime(train_data.loc[:,'ClaimStartDt'], dayfirst=True)
train_data['ClaimEndDt'] = pd.to_datetime(train_data.loc[:,'ClaimEndDt'], dayfirst=True)
train_data['ClaimDays'] = (train_data['ClaimEndDt'] - train_data['ClaimStartDt']).dt.days

eval_data['ClaimStartDt'] = pd.to_datetime(eval_data.loc[:,'ClaimStartDt'], dayfirst=True)
eval_data['ClaimEndDt'] = pd.to_datetime(eval_data.loc[:,'ClaimEndDt'], dayfirst=True)
eval_data['ClaimDays'] = (eval_data['ClaimEndDt'] - eval_data['ClaimStartDt']).dt.days

# Merge Provider Fraud dataset onto beneficiary dataset
train_data = pd.merge(train_data, train_prov, on='ProviderID')
eval_data = pd.merge(eval_data, eval_prov, on='ProviderID')

# train_data.dtypes
# eval_data.dtypes

# print(train_data.isna().sum())
# print(eval_data.isna().sum())

# remove columns (remove IDs, dates, diagnosis/procedure codes, states, county - variables that dont make sense
# being aggregated by provider)
columns_remove = ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt','AttendingPhysician', 'OperatingPhysician',
        'OtherPhysician', 'ClmDiagnosisCode_1','ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
        'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3','ClmProcedureCode_4',
        'ClmAdmitDiagnosisCode', 'AdmissionDt','DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD', 'State', 'County']

train_data_clean = train_data.drop(columns=columns_remove, axis=1)
eval_data_clean = eval_data.drop(columns=columns_remove, axis=1)

# print(train_data_clean.isna().sum())
# print(eval_data_clean.isna().sum())

# remove NA deductibles as 0 exists, and it seems likely they aren't meant to be 0's (low number as well)
train_data_clean = train_data_clean[train_data_clean['DeductibleAmtPaid'].notna()]
eval_data_clean = eval_data_clean[eval_data_clean['DeductibleAmtPaid'].notna()]

# print(train_data_clean.isna().sum())
# print(eval_data_clean.isna().sum())

# convert gender and race to categorical variables, and create dummy variables for the columns
train_data_clean.Gender = train_data_clean.Gender.astype('category')
train_data_clean = pd.get_dummies(train_data_clean, columns=['Gender', 'Race'], drop_first=True)

eval_data_clean.Gender = eval_data_clean.Gender.astype('category')
eval_data_clean=pd.get_dummies(eval_data_clean, columns=['Gender','Race'], drop_first=True)

# train_data_clean.head(2)

# convert fraud yes = 1, no = 0
train_data_clean.Fraud.replace(['Yes', 'No'], ['1', '0'], inplace=True)

# train_data_clean.shape
# eval_data_clean.shape

# aggregate data to providers level
columns_nochangedt = [i for i in train_data_clean.columns if i not in ['ProviderID', 'Fraud']]
for col in columns_nochangedt:
    train_data_clean[col] = pd.to_numeric(train_data_clean[col])
    eval_data_clean[col] = pd.to_numeric(eval_data_clean[col])

train_data_clean_prov_group = train_data_clean.groupby(['ProviderID','Fraud'],as_index=False).agg(['mean']).reset_index()

eval_data_clean_prov_group = eval_data_clean.groupby(['ProviderID'],as_index=False).agg(['mean']).reset_index()

# add in columns for no. of claims, no. of physician, no. of patients, and no. of claims per patient for each provider
###############

extra_columns = train_data.groupby(['ProviderID','Fraud'], as_index=False)['ClaimID','BeneID','AttendingPhysician'].nunique()
extra_columns1 = train_data.groupby(['ProviderID','Fraud'], as_index=False).ClaimID.agg('count').drop('ClaimID', axis = 1)
extra_columns1 = extra_columns1.join(extra_columns)
extra_columns1['PatientClaimNo'] = extra_columns1['ClaimID']/extra_columns1['BeneID']
extra_columns1 = extra_columns1.rename(columns = {'ClaimID':'ClaimNo', 'BeneID':'PatientNo', 'AttendingPhysician':'PhysiNo'}).drop('Fraud', axis = 1)
train_data_clean_prov_group = train_data_clean_prov_group.merge(extra_columns1, on='ProviderID')
train_data_clean_prov_group = train_data_clean_prov_group.drop('ProviderID',axis=1)
train_data_clean_prov_group.rename(columns={train_data_clean_prov_group.columns[0]: "ProviderID", train_data_clean_prov_group.columns[1]: "Fraud"}, inplace = True)

extra_columns_eval = eval_data.groupby(['ProviderID'], as_index=False)['ClaimID','BeneID','AttendingPhysician'].nunique()
extra_columns_eval1 = eval_data.groupby(['ProviderID'], as_index=False).ClaimID.agg('count').drop('ClaimID', axis = 1)
extra_columns_eval1 = extra_columns_eval1.join(extra_columns_eval)
extra_columns_eval1['PatientClaimNo'] = extra_columns_eval1['ClaimID']/extra_columns_eval1['BeneID']
extra_columns_eval1 = extra_columns_eval1.rename(columns = {'ClaimID':'ClaimNo', 'BeneID':'PatientNo', 'AttendingPhysician':'PhysiNo'})
eval_data_clean_prov_group = eval_data_clean_prov_group.merge(extra_columns_eval1, on='ProviderID')
eval_data_clean_prov_group = eval_data_clean_prov_group.drop('ProviderID',axis=1)
eval_data_clean_prov_group.rename(columns={eval_data_clean_prov_group.columns[0]: "ProviderID"}, inplace = True)

# declare predictors and response variables
x_var = train_data_clean_prov_group.drop(axis=1, columns=['ProviderID', 'Fraud'])
y_var = train_data_clean_prov_group['Fraud']

x_var_eval = eval_data_clean_prov_group.iloc[:,1:]

print("BP1")

# Split our data into training and testing sets (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x_var,y_var,test_size=0.3,random_state=1, shuffle = True)
y_train = pd.to_numeric(y_train)
y_test = pd.to_numeric(y_test)
test_train_data = [x_train, x_test, y_train, y_test]

# Print out relevant performance metrics for a particular model (given predictions made using optimal threshold)
def statsummary(test_train_data_curr, y_train_preds, y_test_preds):
    y_train = test_train_data_curr[2]
    y_test = test_train_data_curr[3]
    print('---------- Stats Summary: ----------')
    cm0 = confusion_matrix(y_train, y_train_preds, labels=[1, 0])
    print('Confusion Matrix Train : \n', cm0)
    cm1 = confusion_matrix(y_test, y_test_preds, labels=[1, 0])
    print('Confusion Matrix Test: \n', cm1)
    total0 = sum(sum(cm0))
    total1 = sum(sum(cm1))
    accuracy0 = (cm0[0, 0] + cm0[1, 1]) / total0
    print('Accuracy Train: ', round(accuracy0, 4))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy Test: ', round(accuracy1, 4))
    sensitivity0 = cm0[0, 0] / (cm0[0, 0] + cm0[0, 1])
    print('Sensitivity Train: ', round(sensitivity0, 4))
    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity Test: ', round(sensitivity1,4))
    specificity0 = cm0[1, 1] / (cm0[1, 0] + cm0[1, 1])
    print('Specificity Train: ', round(specificity0,4))
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity Test: ', round(specificity1,4))
    # KappaValue = cohen_kappa_score(y_test, y_test_pred_proba)
    # print("Kappa Value :", round(KappaValue,4))
    print("F1-Score Train: ", round(f1_score(y_train, y_train_preds),4))
    print("F1-Score Test: ", round(f1_score(y_test, y_test_preds),4))


def plot_ROC(fpr,tpr,roc_auc,plot_title):
    fig1 = plt.figure()
    plt.title(plot_title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# Logistic Regression with L2 Penalty (Ridge Regression) (using CV to tune hyper-parameter Lambda)
def lr_statsummary(test_train_data_curr):
    x_train = test_train_data_curr[0]
    x_test = test_train_data_curr[1]
    y_train = test_train_data_curr[2]
    y_test = test_train_data_curr[3]

    log = LogisticRegressionCV(cv=10, class_weight='balanced', random_state=1, max_iter=10000)
    log.fit(x_train, y_train)

    y_train_pred_proba = log.predict_proba(x_train)[::, 1]
    y_test_pred_proba = log.predict_proba(x_test)[::, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    print("Model parameters:")
    print("Model Predictor weights:", log.coef_)  # weights of each feature
    print("Model Intercept:", log.intercept_)  # value of intercept
    print("Model Lambda:", log.C_)  # value of lambda
    print("Model AUC:", round(roc_auc, 4)) # Area under ROC = AUC
    print("Model Optimal Threshold:", round(optimal_threshold,3)) # selected as the threshold that maximises Youden's J statistic

    # make predictions for y based on optimal threshold
    y_train_preds = (y_train_pred_proba > optimal_threshold).astype(bool)
    y_test_preds = (y_test_pred_proba > optimal_threshold).astype(bool)

    # return summary of statistics based on this threshold
    statsummary(test_train_data_curr, y_train_preds, y_test_preds)

    # plot ROC
    plot_ROC(fpr, tpr, roc_auc, 'Receiver Operating Characteristic - Logistic Regression with L2 Penalty')

lr_statsummary(test_train_data)



# Logistic Regression with L1 Penalty (Lasso) (using CV to tune hyper-parameter Lambda)
def las_lr_statsummary(test_train_data_curr):
    x_train = test_train_data_curr[0]
    x_test = test_train_data_curr[1]
    y_train = test_train_data_curr[2]
    y_test = test_train_data_curr[3]

    log_lasso = LogisticRegressionCV(cv=10, class_weight='balanced', random_state=1, max_iter=10000, penalty='l1',
                                     solver='liblinear')
    log_lasso.fit(x_train, y_train)

    y_train_pred_proba = log_lasso.predict_proba(x_train)[::, 1]
    y_test_pred_proba = log_lasso.predict_proba(x_test)[::, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    print("Model parameters:")
    print("Model Predictor weights:", log_lasso.coef_)  # weights of each feature
    print("Model Intercept:", log_lasso.intercept_)  # value of intercept
    print("Model Lambda:", log_lasso.C_)  # value of lambda
    print("Model AUC:", round(roc_auc, 4)) # Area under ROC = AUC
    print("Model Optimal Threshold:", round(optimal_threshold,3)) # selected as the threshold that maximises Youden's J statistic

    # make predictions for y based on optimal threshold
    y_train_preds = (y_train_pred_proba > optimal_threshold).astype(bool)
    y_test_preds = (y_test_pred_proba > optimal_threshold).astype(bool)

    # return summary of statistics based on this threshold
    statsummary(test_train_data_curr, y_train_preds, y_test_preds)

    # plot ROC
    plot_ROC(fpr, tpr, roc_auc, 'Receiver Operating Characteristic - Logistic Regression with L1 Penalty')


las_lr_statsummary(test_train_data)


# K-Nearest Neighbours (using CV to tune hyper-parameter K)
def knn_statsummary(test_train_data_curr):
    x_train = test_train_data_curr[0]
    x_test = test_train_data_curr[1]
    y_train = test_train_data_curr[2]
    y_test = test_train_data_curr[3]

    knn = KNeighborsClassifier(metric='euclidean') # grid search using CV to find the best k between 1 and 25
    k_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(knn, k_grid, cv=10)
    knn_gscv.fit(x_train, y_train)

    y_train_pred_proba = knn_gscv.predict_proba(x_train)[::, 1]
    y_test_pred_proba = knn_gscv.predict_proba(x_test)[::, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    print("Model parameter:")
    print("Model K (neighbours):", knn_gscv.best_params_['n_neighbors'])  # value of lambda
    print("Model AUC:", round(roc_auc, 4)) # Area under ROC = AUC
    print("Model Optimal Threshold:", round(optimal_threshold,3)) # selected as the threshold that maximises Youden's J statistic

    # make predictions for y based on optimal threshold
    y_train_preds = (y_train_pred_proba > optimal_threshold).astype(bool)
    y_test_preds = (y_test_pred_proba > optimal_threshold).astype(bool)

    # return summary of statistics based on this threshold
    statsummary(test_train_data_curr, y_train_preds, y_test_preds)

    # plot ROC
    plot_ROC(fpr, tpr, roc_auc, 'Receiver Operating Characteristic - K-Nearest Neighbours')

knn_statsummary(test_train_data)

# Random Forest (with grid search CV to find best hyper-parameters for RF)
# Will let features other than max_depth and n_estimators to be defaulted
# NB: smaller range of parameters used here as the model takes a long time to run - ideally larger ranges
# of hyper-parameters are used to ensure the best fit for the data
# So far, the random forest has the best results in terms of AUC

def rfc_statsummary(test_train_data_curr):
    x_train = test_train_data_curr[0]
    x_test = test_train_data_curr[1]
    y_train = test_train_data_curr[2]
    y_test = test_train_data_curr[3]

    max_depth = [2, 6]
    n_estimators = [50, 100, 200]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
    rfc = RandomForestClassifier(class_weight='balanced', random_state=1, max_depth=max_depth,
                                 n_estimators=n_estimators)
    rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10)
    rfc_cv.fit(x_train, y_train)

    y_train_pred_proba = rfc_cv.predict_proba(x_train)[::, 1]
    y_test_pred_proba = rfc_cv.predict_proba(x_test)[::, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    print("Model parameters:")
    print("Model n_estimators (number of trees):", rfc_cv.best_params_['n_estimators'])  # value of lambda
    print("Model max_depth:", rfc_cv.best_params_['max_depth'])  # value of lambda
    print("Model AUC:", round(roc_auc, 4))  # Area under ROC = AUC
    print("Model Optimal Threshold:",
          round(optimal_threshold, 3))  # selected as the threshold that maximises Youden's J statistic

    # make predictions for y based on optimal threshold
    y_train_preds = (y_train_pred_proba > optimal_threshold).astype(bool)
    y_test_preds = (y_test_pred_proba > optimal_threshold).astype(bool)

    # return summary of statistics based on this threshold
    statsummary(test_train_data_curr, y_train_preds, y_test_preds)

    # plot ROC
    plot_ROC(fpr, tpr, roc_auc, 'Receiver Operating Characteristic - Random Forest')

    # print feature importance
    print("Random Forest Feature Importance:", rfc_cv.best_estimator_.feature_importances_)

rfc_statsummary(test_train_data)


#
# x_imp_var_eval = x_var_eval.iloc[:,np.r_[imp_indices]]
# imp_rfc_test_pred = imp_rfc.predict(x_imp_var_eval)
# # imp_rfc_test_pred = pd.Dataframe(imp_rfc_test_pred)
# pd.DataFrame(imp_rfc_test_pred).to_csv("aa.csv")
# pd.DataFrame(eval_prov).to_csv("ab.csv")
# imp_rfc_test_pred1 = imp_rfc.predict_proba(x_imp_var_eval)
# pd.DataFrame(imp_rfc_test_pred1).to_csv("ac.csv")
#
#
# eval_data_clean_prov_group.shape
# # random forest - feature importance
# feature_imp = pd.Series(rfc.feature_importances_, x_var.columns).sort_values(ascending=False)
# feature_imp
# feature_imp_index = pd.Series(rfc.feature_importances_).sort_values(ascending=False)
# feature_imp_index
#
# # re-run model with only features above .01 importance (improved rfc)
# imp_indices = [20,28,1,29,0,16,23,31,17,30,4,9,26,15,22,27,3,12,7]
#
# x_imp_var = x_var.iloc[:,np.r_[imp_indices]]
#
# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_imp_var,y_var,test_size=0.3,random_state=1, shuffle = True)
# y_train1 = pd.to_numeric(y_train1)
# y_test1 = pd.to_numeric(y_test1)
# imp_rfc = RandomForestClassifier(class_weight='balanced', random_state=1, max_depth=4)
# imp_rfc.fit(x_train1, y_train1)
# y_train_pred_proba = imp_rfc.predict(x_train1)
# y_test_pred_proba = imp_rfc.predict(x_test1)
#
# cm0 = confusion_matrix(y_train1, y_train_pred_proba, labels=[1, 0])
# print('Confusion Matrix Train : \n', cm0)
#
# cm1 = confusion_matrix(y_test1, y_test_pred_proba, labels=[1, 0])
# print('Confusion Matrix Test: \n', cm1)
#
# total0 = sum(sum(cm0))
# total1 = sum(sum(cm1))
#
# accuracy0 = (cm0[0, 0] + cm0[1, 1]) / total0
# print('Accuracy Train: ', accuracy0)
#
# accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
# print('Accuracy Test: ', accuracy1)
#
# sensitivity0 = cm0[0, 0] / (cm0[0, 0] + cm0[0, 1])
# print('Sensitivity Train : ', sensitivity0)
#
# sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
# print('Sensitivity Test: ', sensitivity1)
#
# specificity0 = cm0[1, 1] / (cm0[1, 0] + cm0[1, 1])
# print('Specificity Train: ', specificity0)
#
# specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
# print('Specificity Test: ', specificity1)
#
# KappaValue = cohen_kappa_score(y_test1, y_test_pred_proba)
# print("Kappa Value :", KappaValue)
#
# AUC = roc_auc_score(y_test1, y_test_pred_proba)
#
# print("AUC         :", AUC)
#
# print("F1-Score Train  : ", f1_score(y_train1, y_train_pred_proba))
#
# print("F1-Score Test  : ", f1_score(y_test1, y_test_pred_proba))
#
# # use improved rfc to predict unseen data
# x_imp_var_eval = x_var_eval.iloc[:,np.r_[imp_indices]]
# imp_rfc_test_pred = imp_rfc.predict(x_imp_var_eval)
# # imp_rfc_test_pred = pd.Dataframe(imp_rfc_test_pred)
# pd.DataFrame(imp_rfc_test_pred).to_csv("aa.csv")
# pd.DataFrame(eval_prov).to_csv("ab.csv")
# imp_rfc_test_pred1 = imp_rfc.predict_proba(x_imp_var_eval)
# pd.DataFrame(imp_rfc_test_pred1).to_csv("ac.csv")
#
# # LDA
#
# def lda_statsummary(y_train, y_test ,x_train, x_test):
#     y_train = pd.to_numeric(y_train)
#     y_test = pd.to_numeric(y_test)
#     lda = LinearDiscriminantAnalysis()
#     lda.fit(x_train, y_train)
#     y_train_pred_proba = lda.predict(x_train)
#     y_test_pred_proba = lda.predict(x_test)
#
# lda_statsummary(y_train, y_test, x_train, x_test)
#
# # QDA
# def qda_statsummary(y_train, y_test ,x_train, x_test):
#     y_train = pd.to_numeric(y_train)
#     y_test = pd.to_numeric(y_test)
#     qda = QuadraticDiscriminantAnalysis()
#     qda.fit(x_train, y_train)
#     y_train_pred_proba = qda.predict(x_train)
#     y_test_pred_proba = qda.predict(x_test)
#
#
# qda_statsummary(y_train, y_test, x_train, x_test)
#
# # Support Vector Machine
# def svm_statsummary(y_train, y_test ,x_train, x_test):
#     y_train = pd.to_numeric(y_train)
#     y_test = pd.to_numeric(y_test)
#     svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#     svm.fit(x_train, y_train)
#     y_train_pred_proba = svm.predict(x_train)
#     y_test_pred_proba = svm.predict(x_test)
#
#
# svm_statsummary(y_train, y_test, x_train, x_test)