# Representing Data and Engineering Features

## Categorical Variables
- one-hot-encoding (or one-out-of-N encoding), which is to replace a categorical variable with one or more new features that can have the values 0 and 1.
- There is a very simple way to encode the data in pandas, using the __get_dummies__ function. The get_dummies function automatically transforms all columns that have object type (like strings) or are categorical into one-hot-encoding.
- This is important to ensure categorical values are represented in the same way in the training set and the test set.
- Be careful: column indexing in pandas includes the end of the range. This is different from slicing a NumPy array, where the end of a range is not included.
- The get_dummies function in pandas treats all numbers as continuous and will not create dummy variables for them. To get around this, you can either use scikit-learn’s __OneHotEncoder__, for which you can specify which variables are continuous and which are discrete, or convert numeric columns in the DataFrame to strings.
