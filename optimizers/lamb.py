import tensorflow as tf


class LAMB(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the LAMB(Layer-wise Adaptive Moments) algorithm.

    Reference: https://arxiv.org/abs/1904.00962
    """
    def __init__(self,
                 learning_rate=.001,
                 beta_1=.9,
                 beta_2=.999,
                 epsilon=1e-6,
                 weight_decay_rate=0.,
                 name='LAMB',
                 **kwargs):

        super(LAMB, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('weight_decay_rate', weight_decay_rate)
        self._set_hyper('decay', self._initial_decay)  # for Keras LR scheduler
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        '''Create slots for the first and second moments.'''
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _prepare_local(self, device, dtype, apply_state):
        super(LAMB, self)._prepare_local(device, dtype, apply_state)
        local_step = tf.cast(self.iterations + 1, dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', dtype))  # make a copy
        beta_2_t = tf.identity(self._get_hyper('beta_2', dtype))
        weight_decay_rate = tf.identity(self._get_hyper('weight_decay_rate', dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        apply_state[(device, dtype)].update(
            dict(
                weight_decay_rate=weight_decay_rate,
                epsilon=tf.convert_to_tensor(self.epsilon, dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        device, dtype = var.device, var.dtype.base_dtype
        coeffs = ((apply_state or {}).get((device, dtype)) or self._fallback_apply_state(device, dtype))

        # update moments
        m = self.get_slot(var, 'm')
        # m_t = β_1 * m_{t−1} + (1 − β_1) * g_t
        m_t = m.assign(coeffs['beta_1_t'] * m + (1 - coeffs['beta_1_t']) * grad, use_locking=self._use_locking)

        # v_t = β_2 * v_{t−1} + (1 − β_2) * {g_t}^2
        v = self.get_slot(var, 'v')
        v_t = v.assign(coeffs['beta_2_t'] * v + (1 - coeffs['beta_2_t']) * tf.square(grad),
                       use_locking=self._use_locking)

        # bias correction
        m_t = m_t / (1. - coeffs['beta_1_power'])
        v_t = v_t / (1. - coeffs['beta_2_power'])

        grad_t = m_t / (tf.sqrt(v_t) + coeffs['epsilon']) + coeffs['weight_decay_rate'] * var

        # calculate ratio
        weight_norm = tf.norm(var, ord=2)
        grad_norm = tf.norm(grad_t, ord=2)
        ratio = tf.where(
            tf.greater(weight_norm, 0),
            tf.where(tf.greater(grad_norm, 0), (weight_norm / grad_norm), 1.),
            1.)

        # update parameters
        var_update = var.assign_sub(ratio * coeffs['lr_t'] * grad_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        device, dtype = var.device, var.dtype.base_dtype
        coeffs = ((apply_state or {}).get((device, dtype)) or self._fallback_apply_state(device, dtype))

        # update moments
        m = self.get_slot(var, 'm')
        # m_t = β_1 * m_{t−1} + (1 − β_1) * g_t
        m_t = m.assign(coeffs['beta_1_t'] * m, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, (1 - coeffs['beta_1_t']) * grad)

        # v_t = β_2 * v_{t−1} + (1 − β_2) * {g_t}^2
        v = self.get_slot(var, 'v')
        v_t = v.assign(coeffs['beta_2_t'] * v, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, (1 - coeffs['beta_2_t']) * tf.square(grad))

        # bias correction
        m_t = m_t / (1. - coeffs['beta_1_power'])
        v_t = v_t / (1. - coeffs['beta_2_power'])

        grad_t = m_t / (tf.sqrt(v_t) + coeffs['epsilon']) + coeffs['weight_decay_rate'] * var

        # calculate ratio
        weight_norm = tf.norm(var, ord=2)
        grad_norm = tf.norm(grad_t, ord=2)
        ratio = tf.where(
            tf.greater(weight_norm, 0),
            tf.where(tf.greater(grad_norm, 0), (weight_norm / grad_norm), 1.),
            1.)

        # update parameters
        var_update = var.assign_sub(ratio * coeffs['lr_t'] * grad_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]
        return tf.group(*updates)

    def get_config(self):
        config = super(LAMB, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decay': self._serialize_hyperparameter('decay'),
            'weight_decay_rate': self._serialize_hyperparameter('weight_decay_rate'),
            'epsilon': self.epsilon,
        })
        return config
