# US Medicare Fraud Analysis
## Overview
In this project, US Medicare data was used to build and assess several predictive models for the purpose of classifying known Healthcare Providers as fraudulent. The best-performing predictive model was then used to analyse unknown Providers to flag as potentially fraudulent.

The following statistical learning methods were considered: 
* Logistic Regression with L2 Regularisation (Ridge Regression)
* Logistic Regression with L1 Regularisation (Lasso)
* K-Nearest Neighbours
* Random Forest
* Improved Random Forest (using feature importance)
* Linear Discriminant Analysis (LDA)
* Quadratic Discriminant Analysis (QDA)
* Support Vector Machine (SVM)

The models are assessed based on their Area under the ROC Curve (AUC) score and any model predictions used are based on the probability threshold that maximises Youden's J statistic (i.e. max(TPR - FPR)). In other words, statistics such as the confusion matrix, accuracy, sensitivity, specificity, and F1-Score are calculated using this set of predictions. Also, where applicable, cross-validation has been applied to tune each of the models' hyper-parameters to ensure an optimal fit.

The Improved Random Forest model was assessed to be the best-performing model in terms of AUC, and was used to produce the ultimate set of predictions for the unknown Providers.

## Input Data
Input data was too large to upload, but can be viewed/downloaded here: https://drive.google.com/drive/folders/1Vovj_PTrj3HF5dXWZ1peZ-eW9vy1W3bG?usp=sharing

### Training Data (used to train the models):
* Medicare_Outpatient_Inpatient_Beneficiary_PartB.csv: Patient claims data consisting of 436,254 individual claims with 55 different features. 
* Medicare_Provider_PartB.csv: Provider data consisting of 4,436 US Medicare Providers and a categorical indicator for fraudulency.

### Evaluation Data (unknown data used by the final model for prediction):
* Medicare_Outpatient_Inpatient_Beneficiary_PartB.csv: Patient claims data consisting of 121,957 individual claims with the same 55 features. 
* Medicare_Provider_PartB.csv: 975 US Medicare Providers with an unknown fraudulency status.

## Results
### Logistic Regression with Ridge Regression:
![image](https://user-images.githubusercontent.com/62014067/127745878-936113bf-4cfc-4e5b-a2df-49f15af53804.png)

### Logistic Regression with Lasso:
![image](https://user-images.githubusercontent.com/62014067/127745910-daef0bfb-1942-4064-8f21-26b4c416c7a6.png)

### K-Nearest Neighbours:
![image](https://user-images.githubusercontent.com/62014067/127745964-218a8eb8-3226-41a3-9f31-56b8406b4774.png)

### Random Forest:
![image](https://user-images.githubusercontent.com/62014067/127745982-2f1a6d95-7431-4157-a951-c0b7471ea371.png)

A feature importance chart was generated for this Random Forest model and the highlighted features were used as the sole predictors in the Improved Random Forest model.

![image](https://user-images.githubusercontent.com/62014067/127746049-0f4fa9ab-caa6-4f3f-b34b-05cc812b9263.png)

### Improved Random Forest:
![image](https://user-images.githubusercontent.com/62014067/127746079-8d02f7a1-d4cf-440b-82c8-35954ce41a92.png)

### LDA:
![image](https://user-images.githubusercontent.com/62014067/127746094-655c6a10-c2f5-4ddf-ab4c-94c0fc54a871.png)

### QDA:
![image](https://user-images.githubusercontent.com/62014067/127746111-491e83da-0b4a-4573-88d0-b7ecc21559ea.png)

### SVM:
![image](https://user-images.githubusercontent.com/62014067/127746139-565c5984-2d8b-4a50-85b1-917bd9cb314f.png)

## Conclusion
The Improved Random Forest performed the best out of all the statistical learning methods with an AUC of 0.9144, and was therefore selected to predict fraudulency status using the unknown dataset.

## Output
Provider_Fraud_Results.csv: Output table consisting of the 975 unknown US Medicare Providers with their predicted fraudulency status and associated fraudulency probabilities.
