# Optimizers for deep learning.

## Notations:
| notation | meaning |
|--------|---------|
| $\theta$ | parameters of the model |
| $u$ | parameter updates |
| $\eta$ | learning rate |
| $g$ or $\nabla_{\theta}J(\theta)$| gradient from the cost function wrt the model parameters |
| $\beta$ | the momentum term |
| $m$ | quantity related to first moment of gradients |
| $v$ | quantity related to higher moment (second for most case) of gradients |
| $T$ | the iteration number |

Here hosted some of the more recent proposed optimizers.  

### Layer-wise Adaptive Rate Scaling
### Yogi
$$
m \leftarrow \beta_1 m + (1 - \beta_1) g \\
v \leftarrow v - (1 - \beta_2)sign(v - g^2) g^2 \\
\hat{\eta} \leftarrow \eta \frac{\sqrt{1 - \beta_2^T}}{1-\beta_1^T} \\
u \leftarrow  - \hat{\eta} \frac{m}{\sqrt{v} + \epsilon} \\
\theta \leftarrow \theta + u
$$

### Rectified Adam
### Layer-wise Adaptive Moments
### Stochastic Weight Averaging
### Lookahead

## Reference on classical optimizers

A very nice review paper by Sebastian Ruder can be found [here](https://arxiv.org/abs/1609.04747), which covers many of the standard optimizers that Tensorflow ships by default.

### Stochastic Gradient Descent.
Update the parameters in the direction of negative gradients.
$$
g \leftarrow \nabla_{\theta}J(\theta) \\
u \leftarrow - \eta g \\
\theta \leftarrow  \theta + u
$$

## SGD with Momentum.
Add a fractioned past gradient updates to the new updates.
$$
g \leftarrow \nabla_{\theta}J(\theta) \\
u \leftarrow \beta u - \eta g \\
\theta \leftarrow \theta + u
$$

## SGD with Nesterov Momentum.
Moving the parameters in previous accumulated gradients direction before evaluating the gradients.
$$
g \leftarrow \nabla_{\theta}J(\theta + \beta u) \\
u \leftarrow \beta u - \eta g \\
\theta \leftarrow \theta + u
$$

## AdaGrad
Accumulating the second moment of gradients, parameter updates are scaled base of square root of it. It works well when the gradients are sparse.
$$
g \leftarrow \nabla_{\theta}J(\theta) \\
v \leftarrow v + g^2 \\
u \leftarrow - \eta \frac{g} {\sqrt{v} + \epsilon} \\
\theta \leftarrow \theta + u
$$


## RMSProp
Use exponential moving average(EMA) of gradient squared as in AdaGrad.
$$
g \leftarrow \nabla_{\theta}J(\theta) \\
v \leftarrow \beta v + (1 - \beta)g^2 \\
u \leftarrow - \eta \frac{g}{\sqrt{v} + \epsilon} \\
\theta \leftarrow \theta + u
$$

## Adadelta
Accumulating both square of gradients as well as square of updates in exponential moving average format, gradient updates are scaled by the ratio between the two.
$$
g \leftarrow \nabla_{\theta}J(\theta) \\
v \leftarrow \beta v + (1 - \beta) g^2 \\
u \leftarrow - \frac{\sqrt{m + \epsilon}}{\sqrt{v + \epsilon}} g \\
m \leftarrow \beta m + (1 - \beta) u^2\\
\theta \leftarrow \theta + u
$$

## Adam
Keep track of exponential moving average of first moment and second moment of the gradients.
Do bias correction on the estimates.
Scale the estimated first moment by the RMS of second moment.
$$
g \leftarrow \nabla_{\theta}J(\theta) \\
m \leftarrow \beta_1 m + (1 - \beta_1) g \\
v \leftarrow \beta_2 v + (1 - \beta_2) g^2 \\
\hat{\eta} \leftarrow \eta \frac{\sqrt{1 - \beta_2^T}}{1-\beta_1^T} \\
u \leftarrow - \hat{\eta} \frac{m}{\sqrt{v} + \epsilon} \\
\theta \leftarrow \theta + u
$$

## AdaMax
Here v is exponentially weighted infinity norm.
$$
g \leftarrow \nabla_{\theta}J(\theta) \\
m \leftarrow \beta_1 m + (1 - \beta_1) g \\
v \leftarrow max(\beta_2 v, \hspace{.2em} abs(g)) \\
u \leftarrow -  \frac{\eta}{1 - \beta_1^T} \frac{m}{v + \epsilon} \\
\theta \leftarrow \theta + u
$$

## Nadam
Adam with Nesterov Momentum.
$$
g \leftarrow\nabla_{\theta} J(\theta - \beta_1 m) \\
m \leftarrow \beta_1 m + (1 - \beta_1) g \\
v \leftarrow \beta_2 v + (1 - \beta_2) g^2 \\
\hat{\eta} \leftarrow \eta \frac{\sqrt{1 - \beta_2^T}}{1-\beta_1^T} \\
u \leftarrow - \hat{\eta} \frac{m}{\sqrt{v} + \epsilon} \\
\theta \leftarrow \theta + u
$$
