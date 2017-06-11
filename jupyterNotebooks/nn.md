# Neural Network

## Multilayer perceptrons (MLP) for classification and regression
- also known as (vanilla) feed-forward neural networks
- can be viewed as generalizations of linear models that perform multiple stages of processing to come to a decision.
- After computing a weighted sum for each hidden unit, a nonlinear function is applied to the result
    + usually the rectifying nonlinearity (also known as rectified linear unit or relu). The relu cuts off values below zero.
    + or the tangens hyperbolicus (tanh), saturates to –1 for low input values and +1 for high input values.
- The default nonlinearity is relu.
- If we want a smoother decision boundary, we could add more hidden units , add a second hidden layer (Figure 2-50), or use the tanh nonlinearity.
- there are many ways to control the complexity of a neural network: 
-   the number of hidden layers
-   the number of units in each hidden layer
-   the regularization (alpha).
- MLPClassifier and MLPRegressor provide easy-to-use interfaces for the most common neural network architectures, they only capture a small subset of what is possible with neural networks.
- Other famous python DL libs:
    + keras
    + lasagna
    + tensor-flow

### Strengths, weaknesses, and parameters
- Given enough computation time, data, and careful tuning of the parameters, neural networks often beat other machine learning algorithms (for classification and regres‐ sion tasks).
- often take a long time to train
- also require careful preprocessing of the data
- Similarly to SVMs, they work best with “homogeneous” data, where all the features have similar meanings.
- For data that has very different kinds of features, tree-based models might work better.
- A common way to adjust parameters in a neural network is to first create a network that is large enough to overfit, making sure that the task can actually be learned by the network. Then, once you know the training data can be learned, either shrink the network or increase alpha to add regularization, which will improve generalization performance.
- we focused mostly on the definition of the model: the number of layers and nodes per layer, the regularization, and the nonlinearity. These define the model we want to learn. There is also the question of how to learn the model, or the algorithm that is used for learning the parameters, which is set using the __algorithm__ parameter:
    + The default is 'adam', which works well in most situations but is quite sensitive to the scaling of the data
    + The other one is 'l-bfgs', which is quite robust but might take a long time on larger models or larger datasets. 
    + There is also the more advanced 'sgd' option, comes with many additional parameters that need to be tuned for best results.
