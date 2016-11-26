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
        * h0, null hypothesis
        * ha, alternative hypothesis
        * level of confidence
        * level of significance
        * binomial test
        * t-test
        * paired t-test
        * 5*2 cross validation (5 time 2 fold)
        * McNemar test (for binary classification)
        * Friedman test
        * Nemenyi post-hoc test

## bias and variance
- bias-variance decomposition
    + generalization error can decompose into bias, variance and noise
    + bias reveals the capability of fitting data
    + variance reveals the effect from fluctuation of data
    + noise reveals the difficulty of the problem trying to solve
    + lack of training data results in poor fitting capability, and fluctuation of data cannot cause big difference, bias is the dominant factor of generalization error
    + increase of training data powers fitting capability and learner can mis-learn the pattern from fluctuation of data, variance becomes the dominant factor.
