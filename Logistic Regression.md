# Logistic Regression

**What is Classification?**
Linear regression assumes that the response variable is a quantitative variable. But in many situations, the response could be a qualitative variable, such as the status of marriage, gender, and so on. Usually, qualitative variables are referred to as categorical variables. *Using statistical learning methods for predicting qualitative responses is called classification*. 
Also, Classification is one of the supervised learning tasks, meaning that training data we feed to algorithms include both features and labels and that we try to predict the classes based on the features. 

when trying to predict classes of response, we usually prefer not to use linear regression. But why?


**Why we do not use linear regression for classification?**
Well, the question should be clarified. Linear regression can be used but there are some limitations.
First, **linear regression can be applied in the case of response with only two classes, that is a binary variable.** In this case, we could naturally encode the two classes as 0 and 1. So we could use a linear regression model to represent the probabilities.
$
p(X) = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p
$
$\beta_0 + \beta_1 X_1 + ... + \beta_p X_p>threshold$ we predict one class
$\beta_0 + \beta_1 X_1 + ... + \beta_p X_p<threshold$ we predict the other class

But when the response has more than two classes, the linear regression is unusable because we cannot find a natural way to convert a qualitative variable with more than two classes into a quantitative variable that is ready for linear regression. For example, suppose that we are trying to predict the medical conditions of a patient in an emergency room based on her symptoms. In the example, we have three possible diagnoses: **stroke, diabetes, and heart disease**. We could consider encoding these values as a quantitative response variable, Y, as follows:
$$
Y = \begin{cases}
1,  \text{if  stroke} \\
2, \text{if  diabetes} \\
3, \text{if  heart disease}
\end{cases}
$$

Using this coding, least squares could be used to fit a linear regression model based on a set of predictors, X1,...,Xp. However, this coding implies an ordering on the outcomes, putting **diabetes* in between **stroke and heart disease,** and insisting that the difference between **diabetes** and **stroke** is the same as the difference between **diabetes** and **heart disease**. In effect, there is no reason that this has to be the case. For instance, one could choose an equally reasonable coding:
$$
Y = \begin{cases}
1,  \text{if  heart disease} \\
2, \text{if  stroke} \\
3, \text{if  diabetes}
\end{cases}
$$
This coding will produce fundamentally different linear models and ultimately lead to a set of different predictions.
On the other hand, if the response is an ordinal categorical, which is a categorical variable with a logical sequence, the encoding of linear regression makes sense. For example, the level of disease can take on a natural ordering, such as *mild, moderate, and severe*, and in this case we felt the gap between mild and moderate was similar to the gap between moderate and severe. So the encoding of 1, 2, 3 is reasonable.


## Logistic Regression
Logistic regression is one of the main algorithms for classification. When using logistic regression, we actually draw upon the method of linear regression to build the relationship between the probability that the response falls into a particular category and predictors. But unlike linear regression, which directly models the $p(Y=1|X)$ and $X$, logistic regression models the relationship between log odds and X.

![Intro to Stats](https://upload-images.jianshu.io/upload_images/10429581-5c8f6e510be78d1d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
The left-side panel is a classifier of linear regression and the right-side panel is a classifier of logistic regression.

**Why do we model log-odds instead of p(x)?**
We can easily find that for balances very close to zero, the probability of default is negative while for very large balances, we probably get a probability of default larger than 1. These predictions are not sensible because the probability of default, regardless of the credit card balance, must fall between 0 and 1. 
To avoid this problem, we usually use the logistic function to model $p(X)$:
$
p(X) = \frac{e^{\beta_0+\beta_1 X}}{1+e^{\beta_0+\beta_1 X}}
$
By fitting a model using the function, we can find that the line is converted into an s-shaped curve. For balances close to zero, the probability will be close to zero but never below it, while for large balances, the probability will be close to 1 but not above it.

**So we know that we need to use the logistic function to build the classifier, but what is the odds? and how does this function come from?**

- *Odds* is the ratio of the probability that an event happens to the probability that an event does not happen.
- Odds ranges from zero to positive infinity.

We know that we model the relationship between log odds and X, and assume that we try to build one univariate classifier, so we have:
$$\log(\frac{p(X)}{1-p(X)}) = \beta_0+\beta_1X$$
Also, we know $$\log(\frac{p(X)}{1-p(X)}) = \log{(odds)}$$
Exponentiate both sides $$\frac{p(X)}{1-p(X)} = e^{\log(odds)}$$
Rearrange the equation a little bit, we have
$$p(X) = \frac{e^{\log(odds)}}{1+e^{\log(odds)}} $$
$$\because \log(\frac{p(X)}{1-p(X)}) = \beta_0+\beta_1X$$
$$\therefore p(X) = \frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}$$

Now, we know the logistic function and how it derivates from log odds. Now we need to estimate the parameters of the logistic function, $\beta_0$, and $\beta_1$.


### Estimating Parameters
In linear regression, the coefficients of $\beta_0$ and $\beta_1$ can be estimated using least squares (which is a fitted line with the minimum sum of residuals). But in logistic regression, the residual can range from negative infinity to positive infinity, so we cannot estimate parameters by residuals.

Instead, we use the *maximum-likelihood*. The general idea of *maximum-likelihood* is to fit a logistic regression model as follows. We try to seek estimates that the predicted probability $p(Y=1|x_i) $ for each observation corresponds to the observed result as closely as possible. In other words, we try to estimate $\beta_0$ and $\beta_1$ such that plugging the two parameters into the model yields a number close to one for all observations with $Y=1$ and a number close to zero for all observations with $Y=0$.
**The idea can be mathematically generalized as follows:**
$$
l(\beta_0,\beta_1) = \prod_{i:y_i=1}p(x_i)\prod_{{i':y_i^{'}=1}p(x_i)}(1-p(x_i))
$$
We try to find two parameters that can maximize the likelihood.


**Now I begin to explain how the process of estimating the parameters using the maximum likelihood works**
Suppose that we have several data of people and want to build a classifier predicting whether they are obese or not based on their weight. We have the following steps:

*Step1: fit a line and project the original data points onto the candidate line, and the projection gives us the log(odds) value* 
(*Note: the fitted line is randomly selected, we will need to model multiple lines like this*)
![image.png](https://upload-images.jianshu.io/upload_images/10429581-c9d1deece7de7362.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Assume that the fitted line has the intercept of -3.84 and slope of 1.83, then we have
$$\log(odds) = -3.84 + 1.83 * Weight$$

*step 2: Transform the log of odds into probabilities using the logistic function.*

Let's take the red point at the left-bottom corner as the demonstration:
![image.png](https://upload-images.jianshu.io/upload_images/10429581-f29221fa42868721.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
We have computed the log of odds for x1, which is -2.1, and then we can get the probability of x1
$$
p(x_1) = \frac{e^{-2.1}}{1+e^{-2.1}} = 0.1
$$
Then we perform the same computation for all data points to attain their probabilities. As a result of this, we convert a straight line to a s-shaped curve constraining the value of response between 0 and 1.
![image.png](https://upload-images.jianshu.io/upload_images/10429581-55cdd9e053c309eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*Step 3: Compute the log of likelihood of the dataset given the curve*
$$
\log(\text{likelihood of dataset}) = \log\prod p(x_i)(1-p(x_i)) \\
=\log(0.49) + \log(0.9) + \log(0.91) + \log(0.91)+ \log(0.92)+\\ 
\log(1-0.01) + \log(1-0.01) + \log(1-0.3) + \log(1-0.9)\\
= -3.77
$$


*Step4: go back to the first step, rotating the fitted line and repeating the setps until we find the maximum log of likelihood*

When we rotate the fitted straight line, we will get different curves of logistic function. Just like the picture shows:
![image.png](https://upload-images.jianshu.io/upload_images/10429581-ab251baf60f8fb09.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Interpretate the Coefficients of Logistic Regression
We also need to know how to interpret the results of logistic regression.

#### X is a continuous variable
Now let's assume that x is a continous variable and we get the following statistics using R.
![image.png](https://upload-images.jianshu.io/upload_images/10429581-33d2d9d294ead582.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- *Estimate*. Intercept estimate means that the log odds of obesity is -3.476 when the weight = 0. This suggests that the observation is not obese when it weighs nothing. Weight estimate = 1.825 means that one-unit increase in weight is associated with an increase in the log odds of obesity by 1.825. 

- *Std Error*. This can measure the accuracy of the coefficient estimates

- *Z value*. 
  $$
  \text{Z statistics} = \frac{coefficient}{std}
  $$
  Z value tells us how many standard deviations is the coefficient estimate away from a standard normal curve with 0 mean and 1 sd. In this example, we can find that z value for intercept is -1.471, meaning that the estimate is 2 less than standard deviations from 0. So we can conclude that the coefficient is not statistically significant. We can get the same conclusion from the coefficient estimate of weight.

- *p value*. P value can help us further confirm whether the coffeicients are statistically significant.

Take another example from *An introduction to stats*

![image.png](https://upload-images.jianshu.io/upload_images/10429581-4acc32865c2f4b04.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   *Coefficient*. Intercept = -10.6513, meaning that the log odds of default is -10.6513 if the person does not have any bank balance. In other words, the leaning method will against you default. On the other hand, the coefficient of balance if 0.0055, meaning that each unit increase in balance is makes the log odds of default rise by 0.0055.

*   *Std. error*. The estimates of standard deviation are rather small, meaning that the estimation of parameters are accurate.

*   *Z-statistics*. Z-statistics of balance is 24.9, suggesting that the coefficient estimate is 24.9 standard deviations away from 0, far more than 2 standard deviations. So there is strong evidence against the null hypothesis that the probability of default does not depend on balance.

*   *P-value.* P-value is far smaller than 0.05, meaning that coefficients are statistically significant. (same as Z-statistics. )
