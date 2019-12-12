EPSILON = 1e-10


def constant(start, end=None, progress=None):
    return start


def linear(start, end, progress):
    return start + progress * (end - start)


def exponential(start, end, progress):
    return start * ((end + EPSILON) / start) ** progress


policies = {
    'constant': constant,
    'linear': linear,
    'exponential': exponential
}


class Scheduler(object):
    def __init__(self, start=1e-3, end=0, total_steps=100, policy='constant'):
        self.start = start
        self.end = end
        self.total_steps = total_steps
        if isinstance(policy, str):
            assert policy in policies, 'not a implemented policy'
            self.policy = policies[policy]
        self.rest()

    def rest(self):
        self.current_step = 0
        self.current_lr = self.start

    def step(self):
        self.current_step += 1
        self.current_lr = self.policy(self.start, self.end, self.current_step / self.total_steps)
        return self.current_lr

    @property
    def done(self):
        return self.current_step > self.total_steps


def _test_scheduler():
    constant_scheduler = Scheduler(policy='constant')
    linear_scheduler = Scheduler(policy='linear')
    exponential_scheduler = Scheduler(policy='exponential')
    schedulers = [constant_scheduler, linear_scheduler, exponential_scheduler]
    while not any(scheduler.done for scheduler in schedulers):
        print(' '.join([str(scheduler.step()) for scheduler in schedulers]))


if __name__ == '__main__':
    # _test_scheduler()
    pass
