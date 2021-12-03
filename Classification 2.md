**A quick review on logistic regression**
Logistic regression tries to model the relationship between predictors and the conditional distribution of the response *Y* given the predictors X using a logistic function. 

$\log{\frac{P(x)}{1-P(X)}} = \beta_0 + \beta_1X_1 + \beta_2X_2+\dots+\beta_pX_p$

logistic regression also has several assumptions
- The predictors are linearly associated with log-odds
- The response is a binary variable with only two classes

To estimate the unknown parameters of the function, $\beta_0, \beta_1,..., \beta_p$,*maximum-likelihood* is used instead of the least-squares. We aim to fit a line that has the maximum likelihood after converting this straight line into an s-shaped curve using the logistic function of log odds.
$l(\beta_0,\beta_1,...,\beta_p) = \prod P(x_i)\prod 1-P(X_i)$

Sometimes, we may need to classify the response with more than two classes; in other words, we need to perform multiclass classification. Although the two-class logistic regression model can be extended to multiple-class extensions, in practice, they tend not to be used very often because another method, *Linear Discriminant Analysis* is more popular for multiple-class classification. 

## Linear Discriminant Analysis and Quadratic Discriminant Analysis
**Agenda**
-  LDA
- QDA
- How to choose between LDA and QDA (variance/bias trade-off)

### LDA

**What is Linear Discriminant Analysis?**
Unlike logistic regression, in which we directly model the conditional distribution of response is given predictors X, LDA indirectly models the conditional distribution of the response by using *Bayes' theorem*. In other words, the method models separately the distribution of predictors X given Y and then uses Bayes theorem to **flip these around** into an estimate for $Pr(Y=k|X=x)$. If the observations are assumed to be normal, then the model is similar to logistic regression model.

**How Bayes Theorem works for LDA?**
Before explaining the use of Bayes Theorem in LDA, let me introduce some terminology about Bayes Theorem first:

**Prior Probability**
Prior probability, in Bayes statistical inference, is the probability of an event before new information or data is collected. It is the best rational assessment of the probability of an outcome based on current knowledge before an experiment is performed.

**Posterior Probability**
Prior probability needs to be revised when new information or data becomes available. The revised probability is *Posterior probability*. In statistical terms, it is the probability of event A occurring given event B occurring.

**Bayes Theorem**
Bayes theorem describes the probability of an event based on new information that is related to the event, that is posterior probability.
$P(A) = \text{the prior probability of A occurring}\\
P(A|B) = \frac{P(AB)}{P(B)} = \frac{P(A)P(B|A)}{P(B)} = \text{the posterior probability of event A given event B occurring}\\
P(B|A) = \frac{P(AB)}{P(A)} = \text{the conditional probability that B occurs given event A}$


Now suppose that we want to classify the observation into one of the k classes, where $k \geq 2$. We let $\pi_k$ represent the prior probability and $f_k(x) = Pr(X=x|Y=k)$ denote the probability distribution that an observation comes from Kth class. We have:
$
Pr(Y=k|X=x) = \frac{Pr(X=x \bigcap Y=k)}{Pr(X=x)} = \frac{Pr(Y=k)Pr(X=x|Y=k)}{Pr(X=x)} = \frac{Pr(Y=k)Pr(X=x|Y=k)}{\sum Pr(Y=K) Pr(X=x|Y=K)}\\
\because \text{we denote }f_k(x) = Pr(X=x|Y=k) \text{ and }\pi_k = Pr(Y=k)\\
\therefore \frac{Pr(Y=k)Pr(X=x|Y=k)}{\sum Pr(Y=K) Pr(X=x|Y=K)} = \frac{\pi_kf_k(x)}{\sum^k_{i=1}\pi_i f_{(x)}}
$
- $Pr(Y=K)$ is the prior probability
- $Pr(Y=k|X=x)$ is the posterior probability
- $Pr(Y=k|X=x) = \frac{Pr(X=x \bigcap Y=k)}{Pr(X=x)} = \frac{Pr(Y=k)Pr(X=x|Y=k)}{Pr(X=x)} $ is Bayes Theorem

#### Estimating parameters of LDA
First let's look at the function of LDA
$\hat{\delta}_k (x) = x\frac{\hat{\mu_k}}{\sigma ^2} - \frac{\hat{\mu}^2_k}{2\sigma^2}+\log(\hat{\pi}_k)$
We do not need to understand and memorize the complex formula. All we need to know is that this is a linear function of x and it has two unknown parameters, $\mu$ and $\sigma$. This is why we call the statistical method **Linear**. 
Based on the formula, we can get the assumption of Linear Discriminant Analysis: *LDA assumes that observations come from Gaussian distribution with a class-specific mean vector and a common variance $\sigma^2$*

The function is a little bit complicated because of Greek Letters. I think it would be better if it is written like this:
$\log(\frac{p(x)}{1-p(x)}) =c_0 + c_1x\\
c_0 \text{ and } c_1 \text{ are functions of } \mu_1,\mu_2, \text{and } \sigma^2.$
In fact, I have written that LDA is just another indirect method of modeling the distribution of response Y compared with logistic regression. So it is very similar to logistic regression and they just differ in the fitting procedure. Therefore, LDA is still modeling the relationship between log odds and predictors, but LDA needs to estimate $\mu$ and $\sigma$ from the sample while logistic regression needs to estimate $\beta_0,..., \beta_p$ using maximum likelihood. 
Therefore, the equation above can be understood as
$
\log(\frac{p(x)}{1-p(x)}) =c_0 + c_1x = x\frac{\hat{\mu_k}}{\sigma ^2} - \frac{\hat{\mu}^2_k}{2\sigma^2}+\log(\hat{\pi}_k)
$
- $c_o$ corresponds to $- \frac{\hat{\mu}^2_k}{2\sigma^2}+\log(\hat{\pi}_k)$
- $c_1$ corresponds to $\frac{\hat{\mu_k}}{\sigma ^2}$

The estimation of the mean and variance of the sample is pretty easy, we just need to calculate them based on the training observations. And we want the response to be the largest when plugging the estimates for mean and variance into the equation.

**OK, now we know a lot about LDA, let me compare it with logistic regression.**
#### Logistic regression Versus LDA
Consider the two-class setting with p=1 estimator, and let $p_1(x)$ and $p_2(x) = 1-p_1(x)$ be the probabilities that x belongs to class 1 and class2.
- Logistic regression can be wirtten as
$\log(\frac{p(x)}{1-p(x)}) = \log(\frac{p_1(x)}{p_2(x)}) = \beta_0+\beta_1x$
- For LDA, the model is
$\log(\frac{p(x)}{1-p(x)}) = \log(\frac{p_1(x)}{p_2(x)})=c_0 + c_1x$
Therefore, based on the functional form of f, we can find that both LDA and logistic regression produce linear decision boundaries.

On the other hand, there is also a significant difference between the two approaches. They have a different method of estimating parameters. 
Logistic regression estimate parameters using *maximum-likelihood*, while linear discriminant analysis estimates parameters, $c_0$ and $c_1$, using the estimated mean and variance from a normal distribution. 

**When to use LDA and when to use logistic regression?**
On the condition that the decision boundary is linear, we should choose between the two approaches based on:
- If the observations come from Gaussian distribution with specific class mean and a common covariance matrix in each class, LDA can provide improvements over logistic regression
- If the normal assumption is not met, then logistic regression performs better than LDA
- If the sample is very small and the response has more than two classes, we also should choose LDA


### Quadratic Discriminant Analysis
LDA assumes that observations within each class are drawn from a multivariate Gaussian distribution with a class-specific mean vector and covariance matrix that is common to all K classes. In contrast, Quadratic discriminant analysis assumes that each class has its own covariance matrix: perhaps the correlation between predictors and class 1 is 0.5 while the correlation between predictors and class 2 is -0.5. And the functional form of QDA is quadratic instead of linear, and this is how quadratic discriminant analysis is called quadratic.

**The function of QDA**
![QDA](https://upload-images.jianshu.io/upload_images/10429581-8cb360904e162a09.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Again we need to understand not the complex equation but that it is a quadratic function of x and it also has two unknown parameters $\mu$ and $\sigma$.

**Summary**
So based on this, we can summarize the similarities and differences between LDA and QDA.
- LDA is a linear function with linear decision boundary, while QDA is a quadratic function with a quadratic decision boundary
- LDA assumes that each class has the same covariance matrix while QDA assumes that each class has its own covariance matrix
- LDA and QDA both assume that observations come from Gaussian distribution


**What are the scenarios where we should LDA and the ones for QDA?**
The question depends on one core concept that is omnipresent in the studying and career of machine learning, *variance and bias tradeoff*. 
- If K classes share the common covariance matrix, the LDA has a linear decision boundary, which means that the coefficients of the LDA model should be linear. In this setting, LDA is a less flexible classifier than QDA and thus has a lower variance. So LDA can improve the performance of predictions. In addition, if the training observations are very small, LDA is also a better option because reducing variance is the priority. On the other hand, QDA will only fit a model much more flexible than necessary, meaning substantially high variance that even cannot be offset by the decrease in bias.
- If the training set is very large or if the assumption of the same covariance matrix is not held, QDA is recommended. In the case of a large train set, the variance reduces as more samples come in and so is not a problem for the model. So we need to pay attention to bias and a more flexible model is needed.

Given that *variance/bias tradeoff* lies at the heart of machine learning and statistical learning, Let me introduce it. As long as we grasp the essence of it, we basically know how to choose among various machine learning algorithms.

#### Variance/Bias tradeoff
We know that the performance of model can be evaluated by MSE(mean squared error), given by:
$MSE = \frac{1}{n}\sum^n_{i=1}(y_i-\hat{f(x_i)})^2$
We want the MSE to be as small as possible because small mse means that the predicted value is close to true value.

Also, we know that the evaluation of the model should be performed on a test set rather than train set. That is to say we finally want to have a model with the minimum test mse.
Now the test MSE can be decomposed into the sum of three parts:
$\text{the variance of }\hat{f}(x_0), \text{the squared bias of }\hat{f}(x_0) \text{ and the variance of the error term } \epsilon.\\
E(y-\hat{f}(x_0))^2 = Var(\hat{f}(x_0))+[\text{Bias}(\hat{f}(x_0))]^2+Var(\epsilon)$

***The equation tells us that in order to minimize the expected test error, we need to select a statistical learning method that simultaneously achieves low variance and low bias.***

Let me quote one example figure from *Introduction to Stats Learning* for an explanation.
![demo](https://upload-images.jianshu.io/upload_images/10429581-55b0f58b6c615fb4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**First, what is variance?**
*Variance* refers to the amount by which *f* would change if we estiamted it using different training data set. Since the training data are used to fit the statistical learning method, different training data sets will result in a different f. But ideally, the estimator for f should not vary too much between training sets. However, if a method has high variance, then a small change in training data can cause large changes in f. Takes the green curve from the left-hand panel of the figure as an example, the green curve is following the data very closely, which means that the predicted values are very close to the actual value. So this curve has a low bias but high variance. 
Why does it have high variance? The reason is that changing any data points may cause the estimate of f to change considerably. We can conclude this by observing the slope of the green curve.

**What is bias?**
*Bias* refers to the error that is introduced by approximating a real-life problem by a much simpler model. For example, linear regression assumes that there is a linear relationship between X and Y. It is unlikely that any real-life problem truly has such a simple relationship, and so performing linear regression will undoubtedly lead to inaccurate predicted values that are far away from true values, that is some bias in f.

**Now with the knowledge of variance and bias, What is variance/bias tradeoff?**
Good test set performance of a statistical learning method requires low variance and low bias. *This is referred to as a trade-off* because it is easy to obtain a method with extremely low bias and high variance (for example we fit a curve that passes through every single observation) or high bias and low variance (we just fit a horizontal line).

**A general rule about v/b tradeoff**
the variance increases but bias decreases as flexibility increases. And the relative rate of change of the two terms determines whether the overall test mse increase ot not. As we increase the flexibility of statistical methods, the bias tends to reduce faster than the variance increases, so the overall test MSE decreases. However, at some point increasing the flexibility has little impact on the reduction in bias but significantly increases variance of model. In this case, the overall test MSE increases. This is why sometimes a simple linear regression model performs better than highly flexible methods, such as KNN and decision trees.


**Exercise**
OK, we know the concept of variance/bias tradeoff, now let me post one exercise about the conception of variance/bias tradeoff.
- *Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The x-axis should represent the amount of flexibility in the method, and the y-axis should represent the values for each curve. There should be five curves. Make sure to label each one.*
- *Explain why?*

**Solution**
![image.png](https://upload-images.jianshu.io/upload_images/10429581-39b0a515f6d00caf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- *Trian Error*. Train error is a monotonically decreasing blue curve because the curve will more closely fit data points as flexibility increases, meaning a more accurate estimate of f.
- *Test Error*. Test error is a U-shaped curve, monotonically decreasing but starting to significantly increase after a particular point. The reason for this is that test error decreases as flexibility increases but at some point, increasing flexibility has little impact on the bias but significantly increases the variance. In this setting, we may have the problem of overfitting, training mse being small but test mse being large. 
- *Bias and Variance*. As a general rule, as flexibility increases, bias will decrease and variance will increase. Variance refers to the amount by which f would change if we estimated it using a different training set. Bias refers to the error that is introduced by approximating a real-life problem by using a simpler model. As flexibility increases, the highly flexible learning methods will follow the data very closely, and in this case they have low bias but high variance because a small change in data points can cause substantial changes in estimates of f.
- *Bayes error*. This is a horizontal line intersecting at the y axis and below Test MSE because the expected test MSE will always be greater the Var(ε*)


**Next article**
- The comparison of various classifiers
- Confusion matrix for the evaluation of the performance of classifiers
