import numpy as np
import pandas as pd
from .scheduler import Scheduler
from matplotlib import pyplot as plt


class LRFinder(object):
    def __init__(self, start_lr=1e-7, max_lr=10., total_steps=100):
        self._history = []
        self.scheduler = Scheduler(start_lr, max_lr, total_steps, 'exponential')

    def step(self, val):
        self._history.append({
            'lr': self.scheduler.current_lr,
            'loss': val,
            'step': self.scheduler.current_step
        })
        return self.scheduler.step()

    @property
    def done(self):
        return self.scheduler.done

    @property
    def history(self):
        return pd.DataFrame(self._history)

    def plot_lr(self, fname):
        assert self.done, 'lr finder not done yet.'
        df = pd.DataFrame(self._history)
        df['log_lr'] = np.log10(df['lr'])
        fig = df.plot(x='log_lr', y='loss')
        plt.savefig(fname)


    # TODO: implement a somewhat reasonable suggestion routine.
    # def suggest_lr(self):
    #     assert self.done, 'lr finder not done yet.'
    #     history = self.history
    #     return history['lr'][history['loss'].diff().idxmax()]


def _test_lr_finder():
    lr_finder = LRFinder()
    while not lr_finder.done:
        lr_finder.step(1)
    # lr_finder.plot_lr()
    print(pd.DataFrame(lr_finder.history))


if __name__ == '__main__':
    _test_lr_finder()
