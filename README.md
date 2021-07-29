# US Medicare Fraud Analysis
In this project, I analyse US Medicare data for the purposes of detecting and predicting Medicare fraud.

I consider the following predictive models: logistic regression with lasso, logistic regression with ridge, k-nearest neighbours, and random forest. In a future update I will add LDA, QDA, SVM, and improved random forest (using feature importance) - the code for this is currently commented out and will be improved on. 

The models are assessed based on their AUC score, and predictions made are based on using the probability threshold that maximises Youden's J statistic (i.e. max(TPR - FPR)).

## Conclusion
Random Forest performs the best out of all the statistical learning methods.
