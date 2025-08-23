from ._base import *



class LR_Base(GradientCalBase):
    est = 0
    def __init__(self, score_fn, target_fn, init=0, iter_fn=None):
        super(LR_Base, self).__init__(init, iter_fn=iter_fn)
        self.psi = score_fn
        self.target_fn = target_fn

    def __call__(self, busy:bool, **kwargs):
        psi = self.psi(**kwargs)
        return super().__call__(psi, busy)
    
    def iterate_est(self, busy:bool, *, weight:float=.05, **kwargs):
        lr = self.__call__(busy, **kwargs)
        return lr * self.target_fn(**kwargs)
    
    def get_est(self, weight:float, *args, **kwargs):
        y = self.iterate_est(*args, **kwargs)
        self.est += weight * (y - self.est)
        return self.est
    


class LR_regenerOL(LR_Base):
    def __init__(self, score_fn, target_fn, init=0, initial_estimate=0, decay_fn=None, *args, **kwargs):
        super(LR_regenerOL, self).__init__(score_fn, target_fn, init, iter_fn=None)
        # Transient observations and cycles counter
        self._cumulator_in_cycle, self._cycle_age, self._n_cycles = 0, 0, 0
        self.gradient_estimate, self.pseudo_estimate = initial_estimate, initial_estimate
        if decay_fn is None:
            self.decay_fn = lambda x: 1/(1+x)
        else:
            self.decay_fn = decay_fn


    def __call__(self, busy:bool, *args, **kwargs):
        lr = super().__call__(busy, **kwargs)
        if busy:
            self._cumulator_in_cycle = self._cumulator_in_cycle + self.target_fn(**kwargs)
            self._cycle_age = self._cycle_age + 1
        else:
            self._cumulator_in_cycle = self.target_fn(**kwargs)
            self._cycle_age = 1
            self._n_cycles = self._n_cycles + 1
        return lr

    def get_est(self, busy:bool, *args, **kwargs):
        lr = self.__call__(busy, *args, **kwargs)
        if not bool(busy):
            self.gradient_estimate = self.pseudo_estimate
        self.pseudo_estimate = self.gradient_estimate + ((self._cumulator_in_cycle - self._cycle_age * kwargs["mean"]) * lr - self._cycle_age * self.gradient_estimate) * self.decay_fn(self._n_cycles)
        return self.pseudo_estimate
    
    def get_estQtlD(self, busy:bool, estDenom, *args, **kwargs):
        lr = self.__call__(busy, *args, **kwargs)
        if not bool(busy):
            self.gradient_estimate = self.pseudo_estimate
        self.pseudo_estimate = self.gradient_estimate + (-(self._cumulator_in_cycle - self._cycle_age * kwargs["mean"]) * lr - estDenom * self.gradient_estimate) * self.decay_fn(self._n_cycles)
        return self.pseudo_estimate