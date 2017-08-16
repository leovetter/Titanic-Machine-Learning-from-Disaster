Notebobook
-------------
In the notebook you can check how logistic regression is applied to the titanic dataset

Training
-------------
You can train your own model on the dataset using the command `python train.py`

Prediction
-------------
Once trained you can generate the predictions file using the command `python predict.py`

Explanation
-------------

At last but not the least below is the complete explanation of how we can implement the logisitic regression function. What we we need is to minimize the cost function J w.r.t to the parameters of the function.
So what we want to compute is $ \frac{\partial{J}}{\partial{w_i}} $ and $ \frac{\partial{J}}{\partial{b}} $ and according to the chaine rule we have $ \frac{\partial{J}}{\partial{w_i}}  = \frac{\partial{J}}{\partial{a}} * \frac{\partial{a}}{\partial{z}} * \frac{\partial{z}}{\partial{w_i}} $ and $ \frac{\partial{J}}{\partial{b}}  = \frac{\partial{J}}{\partial{a}} * \frac{\partial{a}}{\partial{z}} * \frac{\partial{z}}{\partial{b}} $. 
So let's now dive into the math :
$$
\frac{\partial{J}}{\partial{a}} = \frac{\exp(-z)}{(1+\exp(-z))^2} = \frac{\exp(-z) + 1 - 1 }{(1+\exp(-z))^2} = \frac{1 + \exp(-z)}{(1+\exp(-z))^2} - \frac{1}{(1+\exp(-z))^2} = a * (1 - a) (1)
$$
$$
\frac{\partial{a}}{\partial{z}} = \frac{- y}{a} +  \frac{1}{1-a} -  \frac{y}{1-a} = \frac{-y}{a} + \frac{1 - y}{1-a} (2)
$$
$$
\frac{\partial{z}}{\partial{w_i}} =  x_i(3)
$$
$$
\frac{\partial{z}}{\partial{b}} =  1(4)
$$
From equations (1), (2), (3) and (4) we can deduce that 
$$
\frac{\partial{J}}{\partial{w_i}} = a * (1 - a) * (\frac{-y}{a} + \frac{1 - y}{1-a}) * x_i = (-y*(1-a) + a*(1-y)) * x_i = (-y+y*a+a-y*a)*x_i = (a-y)*x_i (5)
$$
and similarly
$$
\frac{\partial{J}}{\partial{b}} =  (a-y) (6)
$$

Once we know the derivatives of the cost function w.r.t the weights we can update the weights with the following rule $w_i = w_i - \alpha * \frac{\partial{J}}{\partial{w_i}} (7)$ and similarly  the rule for the biase is $b = b - \alpha * \frac{\partial{J}}{\partial{b}} (8) $

Concretely equations (5), (6), (7) and (8) are used in the implementation of the logistic regression (cf the file train.py or the notebook)

```
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
```
> Written with [StackEdit](https://stackedit.io/).

