import tensorflow as tf


class Yoji(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Yoji algorithm.

    Yoji is a variant of Adam, which use additve instead of mutilictive updates to the second moment.

    Reference: https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
    """
    def __init__(self,
                 learning_rate=.01,
                 beta_1=.9,
                 beta_2=.999,
                 epsilon=1e-3,
                 name='Yoji',
                 **kwargs):
        super(Yoji, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)  # for Keras LR scheduler
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v', tf.constant_initializer(1.))

    def _resource_apply_dense(self, grad, var):
        dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        beta_1_t = self._get_hyper('beta_1', dtype)
        beta_2_t = self._get_hyper('beta_2', dtype)
        epsilon = tf.convert_to_tensor(self.epsilon, dtype)
        local_step = tf.cast(self.iterations + 1, dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        lr = lr_t * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        m_t = m.assign(beta_1_t * m + (1. - beta_1_t) * grad, use_locking=self._use_locking)
        sign = tf.sign(v - tf.square(grad))
        v_t = v.assign(v - (1 - beta_2_t) * sign * tf.square(grad), use_locking=self._use_locking)

        var_update = var.assign_sub(lr * m_t / tf.sqrt(v_t + epsilon))
        updates = [var_update, m_t, v_t]
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        beta_1_t = self._get_hyper('beta_1', dtype)
        beta_2_t = self._get_hyper('beta_2', dtype)
        epsilon = tf.convert_to_tensor(self.epsilon, dtype)
        local_step = tf.cast(self.iterations + 1, dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        lr = lr_t * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        m_scaled_g_values = (1. - beta_1_t) * grad
        m_t = m.assign(beta_1_t * m, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        sign = tf.sign(tf.gather(v, indices) - tf.square(grad))

        v_scaled_g_values = (1. - beta_2_t) * sign * tf.square(grad)
        v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        var_update = self._resource_scatter_sub(var, indices, lr * m_t / tf.sqrt(v_t + epsilon))
        updates = [var_update, m_t, v_t]
        return tf.group(*updates)


    def get_config(self):
        config = super(Yoji, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decay': self._serialize_hyperparameter('decay'),
            'epsilon': self.epsilon,
            'total_steps': self._serialize_hyperparameter('total_steps'),
        })
        return config



