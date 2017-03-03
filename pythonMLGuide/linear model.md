# Linear model
- Definition, given a dataset with k features, x = (x1;x2;..;xk), linear model is trying to learn a predictive function f based on linear combinations of the k features:
    + f(x) = w1x1 + w2x2 + .. + wkxk + b,
    + in vecter form, f(x) = wTx + b, where w = (w1;w2;..;wk), T means transform.
- Good comprehensibility, the importance level of features show clealy by w.
- Nonlinear models can be derived by the linear models.
- Linear regression
    + euclidean distance
    + least square method
    + parameter estimation
    + multivariate linear regression
- Log-linear regression
    + ln(y) = wTx + b
    + linear model generalization:
        * g(y) = wTx + b, where g(.) has to be continuous and sufficiently smooth
- Logit regression (or logistic regression)
    + classification method
    + unit-step function
    + logistic function:
        * y = 1/(1 + e^(-z))
        * sigmoid function
        * convex function
    + ln(y/(1-y)) = wTx + b
    + no assumption on data distribution
    + can output probability
    + maximum likelihood method
    + gradient descent method
    + newton method
- Linear Discriminant Analysis
    + within-class scatter matrix
    + between-class scatter matrix
    + can be used as a supervised dimension reduction
- Multi-class classification problems can decomposite into multiple binary-class classification problems
    + one vs one (OvO)
        * N(N-1)/2 sub binary classifiers
    + one vs rest (OvR)
        * N sub binary classifiers
    + Many vs Many (MvM)
        * Error Correcting Output Codes (ECOC)
        * encode
        * decode
- class-imbalance
    + imbalance of sample numbers of different classes in the dataset
    + undersampling
        * remove samples to rebalance the dataset
        * EasyEnsemble algorithm
    + oversampling
        * add samples to rebalance the dataset
        * SMOTE
    + threshold-moving
        * rescaling
