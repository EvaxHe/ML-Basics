Serves as the building block for other widely used and complicated machine learning algorithms like Random Forest, XGBoost, and LightGBM. 

## Node Splitting 

A decision tree makes decisions by splitting nodes into sub-nodes. This process is performed multiple times during the training process until only homogenous nodes are left. 

Node splitting techinque based on target variable type 
* Continuous target variable:
  * Reduction in variance 
* Categorical target variable:
  * Gini Impurity 
  * Entropy/Information Gain 
  * Chi-square 
 
 
 ### Splitting Method #1 Reduction in Variance 
 
 ![image](https://user-images.githubusercontent.com/59746522/140235529-b3376747-583f-4709-926d-ae1199dfaf9c.png)
 
1. For each split, individually calculate the variance of each child node
2. Calculate the variance of each split as the weighted average variance of child nodes 
 - (variance* sample size in each child node/total size from parent node) 
  - Do it for each variable (sum up the weighted variance for two subnodes as the splitting result by that class) 

3. Select the split with the lowest variance
4. Perform steps 1-3 until completely homogeneous nodes are achieved (variance = 0) 

### Splitting Method #2 Informationn Gain 

Information Gain = 1 - Entropy 

![image](https://user-images.githubusercontent.com/59746522/140237338-aa5669fa-a1e3-4b14-9bdd-956cde6a0cf8.png)


**Entropy** is use to calculate the purity of the node. Lower the entropy, higher the purity. (homogeneous node: entropy = 0) 

1. For each split, individually calculate the entropy of each child node
2. Calculate the entropy of each split as the weighted average entropy of child nodes
3. Select the split with the lowest entropy or highest information gain
4. Until you achieve homogeneous nodes, repeat steps 1-3

### Splitting Method #3 Gini Impurity 

The most popular and easiest way to split a decision tree, is perferred to Information Gain b/c it does not cotain logarithms which are computationally intensive

* Gini: The probability of correctly labeling a randomly chosen element in that node 

![image](https://user-images.githubusercontent.com/59746522/140237951-7e33f6ed-47be-4dc6-a960-8ad4edf98b90.png)

* Gini Impurity = 1 - Gini 
  * Lower the Gini Impurity, higher the homogeneity 

### Splitting Method #4: Chi-Square 

Works on the statistical significance of differences between the parent node and child nodes.

![image](https://user-images.githubusercontent.com/59746522/140239775-4cbbbbb4-93d4-42ee-844c-2f9761e349b7.png)

Here, the Ecpected is the expected value for a class in a child node (not actual) based on the distribution of classes in the parent node. 

Take the sum of Chi-Square values for all the classes in a node to calculate the Chi-Square for that node, the higher the Chi-Square for that node, the bigger the difference between parent and child nodes. **i.e. higher the homogeneity**




