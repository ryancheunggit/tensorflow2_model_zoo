# Optimizers for deep learning.

## Notations:
| symbol | meaning |
|--------|---------|
| $\theta$ | parameters of the model |
| $u$ | parameter updates |
| $\eta$ | learning rate |
| $J(\theta)$ | cost function wrt the model parameters  |
| $\nabla_{\theta}J(\theta)$ | gradient of the cost function wrt the model parameters |
| $\beta$ | the momentum term |
| $v$ | estimated first moment of gradients |
| $m$ | estimated second moment of gradients |
| $T$ | the iteration number |

Here hosted some of the more recent proposed optimizers.  

### Layer-wise Adaptive Rate Scaling
### Rectified Adam
### Layer-wise Adaptive Moments
### Stochastic Weight Averaging
### Lookahead

## Reference on classical optimizers

A very nice review paper by Sebastian Ruder can be found [here](https://arxiv.org/abs/1609.04747), which covers many of the standard optimizers that Tensorflow ships by default.

### Stochastic Gradient Descent.
Update the parameters directly using the learning rate and gradients.
$$
u \leftarrow - \eta \nabla_{\theta}J(\theta) \\
\theta \leftarrow  \theta + u
$$

## SGD with Momentum.
Add momentum to the updates.
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
Accumulating the square of gradients for parameters, parameter updates are scaled base of it.  
$$
v \leftarrow v + \nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta) \\
u \leftarrow - \eta \nabla_{\theta}J(\theta) \oslash \sqrt{v + \epsilon} \\
\theta \leftarrow \theta + u
$$


## RMSProp
Use exponential moving average of gradient squared in AdaGrad.
$$
v \leftarrow \beta v + (1 - \beta)\nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta) \\
u \leftarrow - \eta \nabla_{\theta}J(\theta) \oslash \sqrt{v + \epsilon} \\
\theta \leftarrow \theta + u
$$

## Adadelta
Accumulating both square of gradients as well as square of updates in exponential moving average format, gradient updates are scaled by the ratio between the two.
$$
v \leftarrow \beta v + (1 - \beta)\nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta) \\
u \leftarrow - \frac{\sqrt{m + \epsilon}}{\sqrt{v + \epsilon}} \nabla_{\theta}J(\theta) \\
m \leftarrow \beta m + (1 - \beta) u \otimes u \\
\theta \leftarrow \theta + u
$$

## Adam
Keep track of exponential moving average of first moment and second moment of the gradients.
Do bias correction on the estimates.
Scale the estimated first moment by the RMS of second moment.
$$
m \leftarrow \beta_1 m + (1 - \beta_1) \nabla_{\theta} J(\theta) \\
v \leftarrow \beta_2 v + (1 - \beta_2) \nabla_{\theta} J(\theta) \otimes \nabla_{\theta} J(\theta) \\
\hat{m} \leftarrow \frac{m}{1 - \beta_1^T} \\
\hat{v} \leftarrow \frac{v}{1 - \beta_2^T} \\
u \leftarrow  - \eta\hat{m} \oslash \sqrt{\hat{v} + \epsilon} \\
\theta \leftarrow \theta + u
$$

## AdaMax
Here v is exponentially weighted infinity norm.
$$
m \leftarrow \beta_1 m + (1 - \beta_1) \nabla_{\theta} J(\theta) \\
v \leftarrow max(\beta_2 v, \hspace{.2em} abs(\nabla_{\theta} J(\theta))) \\
u \leftarrow -  \frac{\eta}{1 - \beta_1^T} \frac{m}{v + \epsilon} \\
\theta \leftarrow \theta + u
$$

## Nadam
Adam with Nesterov Momentum.
$$
m \leftarrow \beta_1 m + (1 - \beta_1) \nabla_{\theta} J(\theta - \beta_1 m) \\
$$
