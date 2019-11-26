import tensorflow as tf


class SWA(tf.keras.optimizers.Optimizer):
    """This class wraps an optimizer to implement SWA(Stochastic Weight Averaging).

    Reference: https://arxiv.org/abs/1803.05407.
    """
    def __init__(self,
                 optimizer,
                 swa_start=0,
                 swa_freq=5,
                 name='SWA',
                 **kwargs):
        super(SWA, self).__init__(name, **kwargs)
        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer), 'Not a valid optimizer'
        assert swa_freq >= 1, 'swa freq should be greater than 1'
        self._optimizer = optimizer
        self._set_hyper('swa_start', swa_start)
        self._set_hyper('swa_freq', swa_freq)
        self._initialized=False

    def _apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations
        return super(SWA, self).apply_gradients(grads_and_vars, name)

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, 'swa')

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


    def _average_op(self, var):
        swa_var = self.get_slot(var, 'swa')
        swa_start = self._get_hyper('swa_start', tf.int64)
        swa_freq = self._get_hyper('swa_freq', tf.int64)
        should_swa_start = tf.greater_equal(self.iterations, swa_start)
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - swa_start, swa_freq))
        should_snapshot_taken = tf.equal(
            self.iterations,
            swa_start + num_snapshots * swa_freq)
        num_snapshots = tf.cast(num_snapshots, tf.float32)
        should_swa_update = tf.reduce_all([should_swa_start, should_snapshot_taken])
        swa_var_update = (swa_var * num_snapshots + var) / (num_snapshots + 1.)
        with tf.control_dependencies([swa_var_update]):
            swa_var_update = swa_var.assign(
                tf.where(should_swa_update, swa_var_update, swa_var),
                use_locking=self._use_locking)
        return swa_var_update

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)
        with tf.control_dependencies([train_op]):
            swa_op = self._average_op(var)
        return tf.group(train_op, swa_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        with tf.control_dependencies([train_op]):
            swa_op = self._average_op(var)
        return tf.group(train_op, swa_op)

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'swa_start': self._serialize_hyperparameter('swa_start'),
            'swa_freq': self._serialize_hyperparameter('swa_freq'),
            }
        base_config = super(SWA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def assign_swa_weights(self, var_list):
        overrides = []
        for var in var_list:
            if 'moving_mean' not in var.name and 'moving_variance' not in var.name:
                var.assign(self.get_slot(var, 'swa'))
            if 'moving_mean' in var.name:
                var.assign(tf.zeros_like(var, var.dtype))
            if 'moving_variance' in var.name:
                var.assign(tf.ones_like(var, var.dtype))
            overrides.append(var)
        print('model weights were updated with SWA weights, BN stats were reseted, run an additional pass to update')
        return tf.group(overrides)
