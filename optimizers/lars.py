import logging
import tensorflow as tf
from tensorflow.python.training import training_ops


class LARS(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the LARS(Layer-wise Adaptive Rate Scaling) algorithm.

    Reference: https://arxiv.org/abs/1708.03888
    """
    def __init__(self,
                 learning_rate=.001,
                 momentum=.9,
                 weight_decay=1e-4,
                 eeta=1e-3,
                 epsilon=1e-8,
                 name='LARS',
                 skip_patterns=None,
                 use_nesterov=False,
                 **kwargs):
        super(LARS, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('momentum', momentum)
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('eeta', eeta)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self._skip_patterns = skip_patterns
        self._use_nesterov = use_nesterov

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'momentum')

    def compute_lr(self, grad, var):
        dtype = var.dtype.base_dtype
        eeta = self._get_hyper('eeta', dtype)
        weight_decay = self._get_hyper('weight_decay', dtype)
        scaled_lr = self._get_hyper('learning_rate', dtype)
        epsilon = tf.convert_to_tensor(self.epsilon, dtype)
        if self._skip_patterns is None or not any(pattern in var.name for pattern in self._skip_patterns):
            weight_norm = tf.norm(var, ord=2)
            grad_norm = tf.norm(grad, ord=2)
            ratio = tf.where(
                    tf.greater(weight_norm, 0),
                    tf.where(
                        tf.greater(grad_norm, 0),
                        eeta * weight_norm / (grad_norm + weight_decay * weight_norm + epsilon),
                        1.),
                    1.)
            scaled_lr *= ratio
            grad = grad + weight_decay * var
        return scaled_lr, grad

    def _resource_apply_dense(self, grad, var):
        dtype = var.dtype.base_dtype
        scaled_lr, grad = self.compute_lr(grad, var)
        mom = self.get_slot(var, 'momentum')
        momentum = self._get_hyper('momentum', dtype)
        return training_ops.resource_apply_momentum(
            var.handle,
            mom.handle,
            tf.cast(1, dtype),
            grad * scaled_lr,
            momentum,
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov)

    def _resource_apply_sparse(self, grad, var, indices):
        logging.info('fallback to momentum optimizer for sparse tensors')
        dtype = var.dtype.base_dtype
        learning_rate = self._get_hyper('learning_rate', dtype)
        mom = self.get_slot(var, 'momentum')
        momentum = self._get_hyper('momentum', dtype)
        return training_ops.resource_apply_momentum(
            var.handle,
            mom.handle,
            learning_rate,
            grad,
            indices,
            momentum,
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov)

    def get_config(self):
        config = super(LARS, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'momentum': self._serialize_hyperparameter('momentum'),
            'decay': self._serialize_hyperparameter('decay'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self._serialize_hyperparameter('epsilon')
        })
        return config
