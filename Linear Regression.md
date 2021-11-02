## Best fit line 

Is obtained by minimzing the cost function **MSE** (using Gradient Descent to update the coefficients)

## Evaluation Metrics for Linear Regression 

* R-squared (Coefficiencts determination)
  * The amout of variation that's explained by the model 
  * Range from 0 to 1 (The higher, the better) 
  * R-squared = 1 - (RSS/TSS)
* Root Mean Squared Error (RMSE) and Residual Standard Error (RSS)
  * RMSE: sqaure root of the variance of the residuals
  * RSS: replce the sample size n in RMSE with degree of freedom (n-2)

R-squared is a better measurement than RSME, b/c the value of RMSE depends on the unites of the variables (it's not a normalized measure) 

## Assumptions: 

* Linearity: Linear relationship between the dependent variable and independent variables 
* Independent of residuals: No correlation between the residual terms
![image](https://user-images.githubusercontent.com/59746522/139966392-087f9fff-24cf-4367-9784-284d2d7f6c4a.jpeg)
* Normal distribution of residuals 
* Homoscedasticity: residuals have constant variance
  * Not constant variance arises in the presence of outliers 

* No or little multicollinearity in multiregression problem 
  * Can be detectede using:
    * Pairwise correlations
    * VIF (Variance inflation factor) 1/(1-Rj^2) - cutoff line:5
  * Solution 
    * Drop some predictors 
    * Standralzied the perdictors by substracting mean 
    * PCA: will squeeze maximum possible information in the first component and then the maximum remaining information in the second component and so on. The primary limitation of this method is the interpretability of the results as the original predictors lose their identity and there is a chance of information loss. 

## Hypothesis Testing 

Is the beta coefficient explain the variance in the data?

* H0: b1 = 0
* H1: b1 != 0 or b1 > 0 

Use t-test - see my demo in python for more details.
