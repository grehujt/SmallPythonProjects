# Representing Data and Engineering Features

## Categorical Variables
- one-hot-encoding (or one-out-of-N encoding), which is to replace a categorical variable with one or more new features that can have the values 0 and 1.
- There is a very simple way to encode the data in pandas, using the __get_dummies__ function. The get_dummies function automatically transforms all columns that have object type (like strings) or are categorical into one-hot-encoding.
- This is important to ensure categorical values are represented in the same way in the training set and the test set.
- Be careful: column indexing in pandas includes the end of the range. This is different from slicing a NumPy array, where the end of a range is not included.
- The get_dummies function in pandas treats all numbers as continuous and will not create dummy variables for them. To get around this, you can either use scikit-learn’s __OneHotEncoder__, for which you can specify which variables are continuous and which are discrete, or convert numeric columns in the DataFrame to strings.


## Binning, Discretization for Linear Models and Trees
- The best way to represent data depends not only on the semantics of the data, but also on the kind of model you are using. 
- Binning features generally has no ben‐ eficial effect for tree-based models, as these models can learn to split up the data any‐ where. In a sense, that means decision trees can learn whatever binning is most useful for predicting on this data.
- __np.digitize__
- The OneHotEncoder does the same encoding as pandas.get_dummies, though it currently only works on categorical variables that are integers.
- Binning features generally has __no beneficial effect for tree-based models__, as these models can learn to split up the data any‐ where. In a sense, that means decision trees can learn whatever binning is most useful for predicting on this data.
- the linear model __benefited greatly__ in expressiveness from the binning  transformation of the data.

## Interactions and Polynomials
- If there are good reasons to use a linear model for a particular dataset, say, because it is very large and high-dimensional, but some features have nonlinear relations with the output, binning can be a great way to increase modeling power.
- Using polynomial features together with a linear regression model yields the classical model of polynomial regression, PolynomialFeatures.
- Using a more complex model, a kernel SVM, is able to learn a similarly complex prediction to the polynomial regression without an explicit transformation of the features.
- he interactions and polynomial features gave us a good boost in perfor‐ mance when using Ridge. Without additional features, the random forest beats the performance of Ridge. Adding interactions and polynomials actually decreases random forest performance slightly.


