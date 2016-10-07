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
    + 别被它的名字迷惑了！这是一个分类算法而不是一个回归算法。该算法可根据已知的一系列因变量估计离散数值（比方说二进制数值 0 或 1 ，是或否，真或假）。简单来说，它通过将数据拟合进一个逻辑函数来预估一个事件出现的概率。因此，它也被叫做逻辑回归。因为它预估的是概率，所以它的输出值大小在 0 和 1 之间（正如所预计的一样）。
    + 假设你的朋友让你解开一个谜题。这只会有两个结果：你解开了或是你没有解开。想象你要解答很多道题来找出你所擅长的主题。这个研究的结果就会像是这样：假设题目是一道十年级的三角函数题，你有 70%的可能会解开这道题。然而，若题目是个五年级的历史题，你只有30%的可能性回答正确。这就是逻辑回归能提供给你的信息。
    + 从数学上看，在结果中，几率的对数使用的是预测变量的线性组合模型。
    
    ```
    odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
    ln(odds) = ln(p/(1-p))
    logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
    ```

    + 在上面的式子里，p 是我们感兴趣的特征出现的概率。它选用使观察样本值的可能性最大化的值作为参数，而不是通过计算误差平方和的最小值(就如一般的回归分析用到的一样)。现在你也许要问了，为什么我们要求出对数呢？简而言之，这种方法是复制一个阶梯函数的最佳方法之一。我本可以更详细地讲述，但那就违背本篇指南的主旨了。

    ```python
    #Import Library
    from sklearn.linear_model import LogisticRegression
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create logistic regression object
    model = LogisticRegression()
     
    # Train the model using the training sets and check score
    model.fit(X, y)
    model.score(X, y)
     
    #Equation coefficient and Intercept
    print('Coefficient: n', model.coef_)
    print('Intercept: n', model.intercept_)
     
    #Predict Output
    predicted= model.predict(x_test)
    ```

    ![pic](pics/logit.jpg)

- 决策树
    + 监督式学习算法
    + 被用于分类问题
    + 同时适用于离散变量和连续变量
    + [A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
    + Advantages:
        * Easy to Understand & Interpret
        * Useful in Data exploration
            Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables.
        * Less data cleaning required
        * It is not influenced by outliers and missing values to a fair degree.
        * Data type is not a constraint: It can handle both numerical and categorical variables.
        * Non Parametric Method: decision trees have no assumptions about the space distribution and the classifier structure.
    + Disadvantages:
        * Overfitting
            Solutions:
            - Setting constraints on tree size
                + Minimum samples for a node split
                + Minimum samples for a terminal node (leaf)
                + Maximum depth of tree (vertical depth)
                + Maximum number of terminal nodes
                + Maximum features to consider for split
            - Tree pruning
                + Unlike greedy setting constraints approach, if we use pruning, we in effect look at a few steps ahead and make a choice.
                + sklearn’s decision tree classifier does not currently support pruning.
        * Not fit for continuous variables
    + When to use:
        * If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.
        * If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
        * If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!

    ```python
    #Import Library
    #Import other necessary libraries like pandas, numpy...
    from sklearn import tree
     
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create tree object 
    model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
     
    # model = tree.DecisionTreeRegressor() for regression
    # Train the model using the training sets and check score
    model.fit(X, y)
    model.score(X, y)
     
    #Predict Output
    predicted= model.predict(x_test)
    ```

- SVM
- 朴素贝叶斯
- K最近邻算法
- K均值算法
- 随机森林算法
- 降维算法
- Gradient Boost 和 Adaboost 算法

**Reference**
- [10 种机器学习算法的要点](http://blog.jobbole.com/92021/)
- [complete-tutorial-tree-based-modeling-scratch-in-python](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
