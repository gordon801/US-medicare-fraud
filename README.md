# US Medicare Fraud Analysis
In this project, I analyse US Medicare data for the purposes of detecting and predicting Medicare fraud.

I consider the following predictive models: logistic regression with lasso, logistic regression with ridge, k-nearest neighbours, and random forest. In a future update I will add LDA, QDA, SVM, and improved random forest (using feature importance) - the code for this is currently commented out and will be improved on before being released. 

The models are assessed based on their AUC score, and predictions made are based on using the probability threshold that maximises Youden's J statistic (i.e. max(TPR - FPR)).

## Input Data
Input data was too large to upload, but can be viewed/downloaded here: https://drive.google.com/drive/folders/1Vovj_PTrj3HF5dXWZ1peZ-eW9vy1W3bG?usp=sharing

## Conclusion
Random Forest performs the best out of all the statistical learning methods with an AUC of 0.90.

![image](https://user-images.githubusercontent.com/62014067/127531245-ee882c3d-0257-434c-8bb7-441562433e68.png)


*(Other scores to be added).*
