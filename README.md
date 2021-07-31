# US Medicare Fraud Analysis
## Overview
In this project, US Medicare data (on patient claims and Provider fraudulency) is used to build and assess several predictive models for the purpose of classifying Providers as fraudulent. 

The following statistical learning methods are considered: 
* Logistic Regression with L2 Regularisation (Ridge Regression)
* Logistic Regression with L1 Regularisation (Lasso)
* K-Nearest Neighbours
* Random Forest
* Improved Random Forest (using feature importance)
* Linear Discriminant Analysis (LDA)
* Quadratic Discriminant Analysis (QDA)
* Support Vector Models

Where applicable, cross-validation has been applied to tune each of the models' hyper-parameters to ensure an optimal fit. The models are assessed based on their AUC score and any model predictions used are based on the probability threshold that maximises Youden's J statistic (i.e. max(TPR - FPR)). In other words, statistics such as the confusion matrix, accuracy, sensitivity, specificity, and F1-Score are calculated using this set of predictions.

The Improved Random Forest model was assessed to be the best overall model in terms of AUC, and was used to produce the ultimate set of predictions for the unknown providers.

## Input Data
Input data was too large to upload, but can be viewed/downloaded here: https://drive.google.com/drive/folders/1Vovj_PTrj3HF5dXWZ1peZ-eW9vy1W3bG?usp=sharing

Training Data (used to train the models):
* Medicare_Outpatient_Inpatient_Beneficiary_PartB.csv: Patient claims data consisting of 436,254 individual claims with 55 different features. 
* Medicare_Provider_PartB.csv: Provider data consisting of 4,436 US Medicare Providers and a categorical indicator for fraudulency.

Evaluation Data (unknown data used by the final model for prediction):
* Medicare_Outpatient_Inpatient_Beneficiary_PartB.csv: Patient claims data consisting of 121,957 individual claims with the same 55 features. 
* Medicare_Provider_PartB.csv: 975 US Medicare Providers with an unknown fraudulency status.

## Results



## Conclusion
Random Forest performed the best out of all the statistical learning methods with an AUC of 0.90.

![image](https://user-images.githubusercontent.com/62014067/127531245-ee882c3d-0257-434c-8bb7-441562433e68.png)
