import numpy as np
from typing import Dict, List, Tuple, Optional

import tensorflow as tf

from continualworld.sac.sac import SAC
from continualworld.utils.utils import reset_optimizer

class CRP(object):
    """
    CRP class used in SLLRL, converted to tensorflow
    taken from https://github.com/HeyuanMingong/sllrl/blob/master/crp.py
    """

    def __init__(self, zeta:float = 1.0):
        super(CRP, self).__init__()
        ## concentration param
        self._zeta = zeta
        ## number of non-empty clusters
        self._L = 1
        ## time period - whats this?
        self._t = 2
        ### prior_dist
        self._prior = np.array([1/(1+zeta), zeta/(1+zeta)])

    def select(self) -> int:
        index = np.random.choice(1 + np.arange(self._L + 1), p = self._prior)
        return index

    def update(self, index:int) -> None:
        self._t += 1 # incremeted each update
        if index == self._L + 1:
            print('A new cluster is added...')
            self._prior = np.concatenate((self._prior, np.zeros(1)), axis=0)
            self._prior[-1] = self._zeta / (self._t - 1 + self._zeta)
            self._prior[-2] = 1 / (self._t - 1 + self._zeta)
            for idx in range(self._L):
                self._prior[idx] *= (self._t - 2 + self._zeta) / (self._t - 1 + self._zeta)
            self._L += 1
        else:
            print('Using existing cluster...')
            for idx in range(self._L + 1):
                if idx == index - 1:
                    self._prior[idx] = ((self._t - 2 + self._zeta)* self._prior[idx]+1) / (self._t-1 + self._zeta)
                else:
                    self._prior[idx] *= (self._t-2+self._zeta) / (self._t - 1 + self._zeta)


class SLLRL_SAC(SAC):
    
    def __init__(
        self,**vanilla_sac_kwargs
    ) -> None:
        """
        SLLRL: https://arxiv.org/abs/2205.10787
        """
        super().__init__(**vanilla_sac_kwargs)
        pass

    def on_test_start(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_test_end(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_task_start(self, current_task_idx: int) -> None:
        ## TODO:
        ## update CRP prior
        ## select agent to learn
        ## NOTE: seems too infrequent and too convenient to do this on task start
        ## ideally would like to do this each update?

        ### CRP ###
        # L = self.crp._L
        # prior = self.crp._prior
        pass

    def on_task_end(self, current_task_idx: int) -> None:
        pass

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        return None