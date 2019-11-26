import tensorflow as tf


class RAdam(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the RAdam(Rectified) algorithm.

    RAdam is a variant of Adam, which introduced a term to rectify the variance of the adaptive learning rate.

    Reference: https://arxiv.org/abs/1908.03265
    """
    def __init__(self,
                 learning_rate=.001,
                 beta_1=.9,
                 beta_2=.999,
                 epsilon=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 sma_threshold=5.,
                 total_steps=0.,
                 warmup_proportion=.1,
                 min_lr=0.,
                 name='RAdam',
                 **kwargs):
        """Construct a new RAdam optimizer.

        Args:
            learning_rate: The learning rate.
            beta_1: exponential decay rate for 1st moments.
            beta_2: exponential decay rate for 2nd moments.
            epsilon: constant for numerical stability.
            weight_decay: weight decay for parameters.
            amsgrad: whether to apply AMSGrad or not.
            sma_threshold: threshold for simple mean average.
            total_steps: total number of training steps, needs to be positive to enable warmup.
            warmup_proportion: the proportion of increate in learning rate during warmup.
            min_lr: minimal learning rate after the warmup.
            name: name for the operations when using optimizer.
            **kwargs: additional args like `clipnorm`, 'lr' etc.
        """
        super(RAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)  # for Keras LR scheduler
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('sma_threshold', sma_threshold)
        self._set_hyper('total_steps', float(total_steps))
        self._set_hyper('warmup_proportion', warmup_proportion)
        self._set_hyper('min_lr', min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self._initial_weight_decay = weight_decay
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        '''Create slots for the first and second moments.'''
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def set_weights(self, weights):
        params = self.weights
        # for compatibility with V1 optimizer
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(RAdam, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var):
        dtype = var.dtype.base_dtype

        lr_t = self._decayed_lr(dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        sma_threshold = self._get_hyper('sma_threshold', dtype)
        beta_1_t = self._get_hyper('beta_1', dtype)
        beta_2_t = self._get_hyper('beta_2', dtype)
        epsilon = tf.convert_to_tensor(self.epsilon, dtype)
        local_step = tf.cast(self.iterations + 1, dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)


        if self._initial_total_steps > 0:
            total_steps = self._get_hyper('total_steps', dtype)
            warmup_steps = total_steps * self._get_hyper('warmup_proportion', dtype)
            min_lr = self._get_hyper('min_lr', dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2. / (1. - beta_2_t) - 1.  # maximum length of the approximated SMA
        sma_t = sma_inf - 2. * local_step * beta_2_power / (1. - beta_2_power)

        # update moments
        m_t = m.assign(beta_1_t * m + (1. - beta_1_t) * grad, use_locking=self._use_locking)
        v_t = v.assign(beta_2_t * v + (1. - beta_2_t) * tf.square(grad), use_locking=self._use_locking)

        # compute bias corrected moving average
        m_corr_t = m_t / (1. - beta_1_power)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1. - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1. - beta_2_power))

        # compute the variance rectification term
        r_t = tf.sqrt((sma_t - 4.) / (sma_inf - 4.) * (sma_t - 2.) / (sma_inf - 2.) * sma_inf / sma_t)

        # update parameters, if variance is tractable, update with adaptive momentum, otherwise with un-adapted.
        grad_t = tf.where(sma_t >= sma_threshold, r_t * m_corr_t / (v_corr_t + epsilon), m_corr_t)
        if self._initial_weight_decay > 0.:
            grad_t += self._get_hyper('weight_decay', dtype) * var
        var_update = var.assign_sub(lr_t * grad_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        dtype = var.dtype.base_dtype

        lr_t = self._decayed_lr(dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        sma_threshold = self._get_hyper('sma_threshold', dtype)
        beta_1_t = self._get_hyper('beta_1', dtype)
        beta_2_t = self._get_hyper('beta_2', dtype)
        epsilon = tf.convert_to_tensor(self.epsilon, dtype)
        local_step = tf.cast(self.iterations + 1, dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)


        if self._initial_total_steps > 0:
            total_steps = self._get_hyper('total_steps', dtype)
            warmup_steps = total_steps * self._get_hyper('warmup_proportion', dtype)
            min_lr = self._get_hyper('min_lr', dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2. / (1. - beta_2_t) - 1.  # maximum length of the approximated SMA
        sma_t = sma_inf - 2. * local_step * beta_2_power / (1. - beta_2_power)

        # update moments
        m_scaled_g_values = (1. - beta_1_t) * grad
        m_t = m.assign(beta_1_t * m, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (1. - beta_2_t) * tf.square(grad)
        v_t = v.assign(beta_2_t * v, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        # compute bias corrected moving average
        m_corr_t = m_t / (1. - beta_1_power)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1. - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1. - beta_2_power))

        # compute the variance rectification term
        r_t = tf.sqrt((sma_t - 4.) / (sma_inf - 4.) * (sma_t - 2.) / (sma_inf - 2.) * sma_inf / sma_t)

        # update parameters, if variance is tractable, update with adaptive momentum, otherwise with un-adapted.
        grad_t = tf.where(sma_t >= sma_threshold, r_t * m_corr_t / (v_corr_t + epsilon), m_corr_t)
        if self._initial_weight_decay > 0.:
            grad_t += self._get_hyper('weight_decay', dtype) * var

        var_update = self._resource_scatter_add(var, indices, tf.gather(-lr_t * grad_t, indices))

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super(RectifiedAdam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decay': self._serialize_hyperparameter('decay'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'sma_threshold': self._serialize_hyperparameter('sma_threshold'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': self._serialize_hyperparameter('total_steps'),
            'warmup_proportion': self._serialize_hyperparameter('warmup_proportion'),
            'min_lr': self._serialize_hyperparameter('min_lr'),
        })
        return config
