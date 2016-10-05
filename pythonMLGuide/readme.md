# Introduction to Machine Learning in Python

## Three ML approaches
- Supervised Learning
    算法由一个目标变量或结果变量（或因变量）组成。这些变量由已知的一系列预示变量（自变量）预测而来。利用这一系列变量，我们生成一个将输入值映射到期望输出值的函数。这个训练过程会一直持续，直到模型在训练数据上获得期望的精确度。监督式学习的例子有：回归、决策树、随机森林、K – 近邻算法、逻辑回归等。

- Unsupervised Learning
    在这类算法中，没有任何目标变量或结果变量要预测或估计。这个算法用在不同的组内聚类分析。这种分析方式被广泛地用来细分客户，根据干预的方式分为不同的用户组。非监督式学习的例子有：关联算法和 K – 均值算法。

- Reenforced Learning
    这类算法训练机器进行决策。它是这样工作的：机器被放在一个能让它通过反复试错来训练自己的环境中。机器从过去的经验中进行学习，并且尝试利用了解最透彻的知识作出精确的商业判断。 强化学习的例子有马尔可夫决策过程。

## Classic learning algorithms
- 线性回归
    + 线性回归通常用于根据连续变量估计实际数值（房价、呼叫次数、总销售额等）。我们通过拟合最佳直线来建立自变量和因变量的关系。这条最佳直线叫做回归线，并且用 Y= a *X + b 这条线性等式来表示。
    + 线性回归的两种主要类型是一元线性回归和多元线性回归。一元线性回归的特点是只有一个自变量。多元线性回归的特点正如其名，存在多个自变量。找最佳拟合直线的时候，你可以拟合到多项或者曲线回归。这些就被叫做多项或曲线回归。
    
    ```python
    #Import Library
    #Import other necessary libraries like pandas, numpy...
    from sklearn import linear_model
     
    #Load Train and Test datasets
    #Identify feature and response variable(s) and values must be numeric and numpy arrays
    x_train=input_variables_values_training_datasets
    y_train=target_variables_values_training_datasets
    x_test=input_variables_values_test_datasets
     
    # Create linear regression object
    linear = linear_model.LinearRegression()
     
    # Train the model using the training sets and check score
    linear.fit(x_train, y_train)
    linear.score(x_train, y_train)
     
    #Equation coefficient and Intercept
    print('Coefficient: n', linear.coef_)
    print('Intercept: n', linear.intercept_)
     
    #Predict Output
    predicted= linear.predict(x_test)
    ```

    ![pic](./pics/linear_regression.png)

- 逻辑回归
- 决策树
- SVM
- 朴素贝叶斯
- K最近邻算法
- K均值算法
- 随机森林算法
- 降维算法
- Gradient Boost 和 Adaboost 算法

