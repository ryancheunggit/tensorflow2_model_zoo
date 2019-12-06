# Optimizers for deep learning.

A very nice review paper by Sebastian Ruder can be found [here](https://arxiv.org/abs/1609.04747), which covers many of the standard optimizers that Tensorflow ships by default.

Here hosted some of the more recent proposed optimizers.  

## Notations:
| symbol | meaning |
|--------|---------|
| $\theta$ | parameters of the model |
| $\eta$ | learning rate |
| $J(\theta)$ | cost function wrt the model parameters  |
| $\nabla_{\theta}J(\theta)$ | gradient of the cost function wrt the model parameters |
| $\beta$ | the momentum term |
| $u$ | parameter updates |
| $m$ |   |

## Stochastic Gradient Descent.
Update the parameters directly using the learning rate multiply gradients.
$$
u \leftarrow - \eta \nabla_{\theta}J(\theta) \\
\theta \leftarrow  \theta + u$$

## SGD with Momentum.
Keep a exponential moving average of the updates, use it to update parameters.
$$
u \leftarrow \beta u - \eta \nabla_{\theta}J(\theta) \\
\theta \leftarrow \theta + u
$$

## SGD with Nesterov Momentum.
Moving the parameters in previous accumulated gradients direction before evaluating the gradients.
$$
u \leftarrow \beta u - \eta \nabla_{\theta}J(\theta + \beta u) \\
\theta \leftarrow \theta + u
$$

## AdaGrad
Accumulating the square of gradients for parameters in $sg$, parameter updates are scaled based on past update frequencies.  
$$
sg \leftarrow sg + \nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta) \\
u \leftarrow - \eta \nabla_{\theta}J(\theta) \oslash \sqrt{sg + \epsilon} \\
\theta \leftarrow \theta + u
$$


## RMSProp
Use exponential moving average of gradient squared in AdaGrad.
$$
sg \leftarrow \beta sg + (1 - \beta)\nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta) \\
u \leftarrow - \eta \nabla_{\theta}J(\theta) \oslash \sqrt{sg + \epsilon} \\
\theta \leftarrow \theta + u
$$

## Adadelta
Accumulating both square of gradients as well as square of updates in exponential moving average format, gradient updates are scaled by the ratio between the two.
$$
sg \leftarrow \beta sg + (1 - \beta)\nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta) \\
u \leftarrow - \frac{\sqrt{su + \epsilon}}{\sqrt{sg + \epsilon}} \nabla_{\theta}J(\theta) \\
su \leftarrow \beta su + (1 - \beta) u \otimes u \\
\theta \leftarrow \theta + u
$$

## Adam
Keep track of exponential moving average of first moment and second moment of the gradients.
Do bias correction on the estimates.
Scale the estimated first moment by the RMS of second moment.
$$
m \leftarrow \beta_1 m - (1 - \beta_1) \nabla_{\theta} J(\theta) \\
v \leftarrow \beta_2 v + (1 - \beta_2) \nabla_{\theta} J(\theta) \otimes \nabla_{\theta} J(\theta) \\
\hat{m} \leftarrow \frac{m}{1 - \beta_1^T} \\
\hat{v} \leftarrow \frac{v}{1 - \beta_2^T} \\
u \leftarrow  - \eta\hat{m} \oslash \sqrt{\hat{s} + \epsilon} \\
\theta \leftarrow \theta + u
$$
