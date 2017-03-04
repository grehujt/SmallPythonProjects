# Decision tree
- information entropy
- split atrribute selection criteria
    + information gain
        * bia to classes with more samples
    + gain ratio
        * intrinsic value (IV)
        * bia to classes with less samples
    + gini index
    + all these criteria may have big influence in resulting tree structures, but generalization performance is seldom affected
- **ID3** use information gain as the criteria to select attributes
- **C4.5** uses heuristics trick based on information gain and gain ratio
    + find attributes with above-mean information gain as candidates
    + find max gain ratio attribute in the candidates
- **CART** (classification and regression tree) uses gini index
- tree pruning
    + to solve overfitting
    + use testing data to validate if a split is needed
    + pre-pruning
        * less branches
        * more risks in underfitting
    + post-pruning
        * more branches
        * less risks in underfitting
        * better generalization performance
        * more training time
    + pruning has big infuence in generalization performance 
- handle continuous values
    + bi-partition
        * adopted in C4.5
        * sort the N values
        * find N-1 middle points
        * select the max information gain split point from the N-1 middle points
    + unlike discrete attributes, continuous attributes would not be excluded from next iteration of attribute selection
- handle missing values
    + how to select split attribute for dataset with missing values
    + how to handle missing values in selected split attribute
- multi-variate decision tree
    + desicion boundaries of uni-variate decision trees are always axis-parallel
    + desicion boundaries of multi-variate decision trees can be oblique
