# Model Evaluation and Selection

## Empirical error (or training error) and overfitting
- error rate
- accuracy
- error
- training error / empirical error
- testing error
- generalization error
- overfitting (key obstacle)
- underfitting

## Model evaluation
- testing set
- testing error
- hold-out
    + sampling
    + stratified sampling
- cross validation
    + k-fold cross validation
    + repeated p times k-fold cross validation
    + Leave-One-Out
- bootstrapping
    + 1/e = ~36.8% testing data
    + useful on small dataset, resemble learning
- parameter tunning
    + validation set (subset of training set)
- performance measure
    + mean squared error in regression tasks
    + error rate or accuracy in classification tasks
        * in binary classification tasks
            - precision = TP / (TP+FP)
            - recall  = TP / (TP+FN)
            - P-R curve
            - Breaking-Even Point (BEP), where precision=recall
            - F1 measure (harmonic mean of precision and recall) = 2*P*R/(P+R)
            - macro-P / macro-R / macro-F1
            - micro-P / micro-R / micro-F1
    + ROC and AUC (in binary classification tasks)
        * Receiver Operating Characteristics curve
        * TPR, True Positive Rate, = TP/(TP+FN) (same as recall)
        * FPR, False Positive Rate, = FP/(TN+FP)
        * Area Under ROC Curve
    + cost-sensitive error rate and cost curve
        * cost matrix
        * cost curve
    + hypothesis test
