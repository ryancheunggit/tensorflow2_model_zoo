import tensorflow as tf


class Lookahead(tf.keras.optimizers.Optimizer):
    """This class wraps an optimizer to implement Lookahead.

    Reference: https://arxiv.org/abs/1907.08610
    """
    def __init__(self,
                 optimizer,
                 alpha=.5,
                 k=6,
                 name='Lookahead',
                 **kwargs):
        super(Lookahead, self).__init__(name, **kwargs)
        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer), TypeError('Not a valid optimizer')
        assert 0. <= alpha <= 1, ValueError('slow weights step size alpha should be in [0, 1]')
        assert k >= 1, ValueError('synchronization period k should >= 1')

        self._optimizer = optimizer
        self._set_hyper('alpha', alpha)
        self._set_hyper('k', k)
        self._initialized = False

    def apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations
        return super(Lookahead, self).apply_gradients(grads_and_vars, name)

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, 'slow')

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper('learning_rate')

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper('learning_rate', lr)

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper('learning_rate')

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper('learning_rate', learning_rate)

    def _init_op(self, var):
        slow_var = self.get_slot(var, 'slow')
        return slow_var.assign(
            tf.where(
                tf.equal(self.iterations, tf.constant(0, self.iterations.dtype)),
                var,
                slow_var
            ), use_locking=self._use_locking)

    def _lookahead_op(self, var):
        dtype = var.dtype.base_dtype
        slow_var = self.get_slot(var, 'slow')
        alpha = self._get_hyper('alpha', dtype)
        k = self._get_hyper('k', tf.int64)
        local_step = tf.cast(self.iterations + 1, tf.int64)
        step_back = slow_var + alpha * (var - slow_var)
        should_slow_update = tf.equal(local_step, tf.math.floordiv(local_step, k) * k)
        with tf.control_dependencies([step_back]):
            slow_update = slow_var.assign(
                tf.where(should_slow_update, step_back, slow_var),
                use_locking=self._use_locking
            )
            var_update = var.assign(
                tf.where(should_slow_update, step_back, var),
                use_locking=self._use_locking
            )
        return tf.group(slow_update, var_update)

    def _resource_apply_dense(self, grad, var):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_dense(grad, var)
            with tf.control_dependencies([train_op]):
                lookahead_op = self._lookahead_op(var)
        return tf.group(init_op, train_op, lookahead_op)

    def _resource_apply_sparse(self, grad, var, indices):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
            with tf.control_dependencies([train_op]):
                lookahead_op = self._lookahead_op(var)
        return tf.group(init_op, train_op, lookahead_op)

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'alpha': self._serialize_hyperparameter('alpha'),
            'k': self._serialize_hyperparameter('k'),
        }
        base_config = super(Lookahead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
