import numpy as np
import warnings



class QueueBase(object):
    clock = 0
    state = 0
    next_in = 0
    next_out = 0
    delay = 0
    idle = 0
    schedule = []

    def __init__(self, source, server, *, max_T=10000, name:str="", **kwargs):
        self.source = source
        self.server = server
        self.MAX_T = max_T
        self.name = name


    def __call__(self, *args, **kwds):
        """
        next event
        """
        rslt = {}

        if self.next_in <= self.next_out:
            event = self._arrive(*args, **kwds)
            rslt["inter_arrival"], self.next_in = self.source.__call__()
        else:
            event = self._depart(*args, **kwds)
            if self.state <= 0:  self.server.clock = self.next_in
            rslt["service_time"], self.next_out = self.server.__call__()

        if self.clock > self.MAX_T: self.reset()

        return event, self.clock, self.state, rslt
    

    def _arrive(self, *args, **kwds):
        event = 1
        self.state += event
        self.clock = self.next_in
        return event

    def _depart(self, *args, **kwds):
        event = -1
        self.state += event
        self.clock = self.next_out
        return event
    

    def _put(self, *args):
        self.schedule = self.schedule + list(args)

    def _get(self):
        try:
            get = self.schedule.pop(0)
        except: 
            get = np.inf
            warnings.warn("It seems that a service schedule is missing.")
        return get
    

    def reset(self, *, backT=None):
        if backT is None: backT = self.clock
        self.source.clock -= backT
        self.server.clock -= backT
        self.next_in -= backT
        self.next_out -= backT
        self.clock -= backT



class AdmissionCtrl(QueueBase):

    def __init__(
        self,
        source,
        server,
        *,
        max_T:int=10000, 
        begin_busy:bool=False,
        state_load:int=0,
        work_load:float=0.,
        **kwargs
    ):
        super().__init__(source, server, max_T=max_T, **kwargs)

        if begin_busy:
            self.state = state_load
            self.delay = work_load
            self.server.clock += work_load
            self.server.workload += work_load
            if state_load > 1:
                self._put(*tuple(np.sort(work_load*np.random.uniform(size=state_load-1))))
            self._put(work_load, work_load+self.next_out)
        self.state_trace = [[self.clock, self.state]]
        self.service_time, next_out = self.server.__call__(self.delay)
        self._put(next_out)
        self.next_out = self._get()


    def __call__(self):
        """
        next arrival
        """
        rslt = {}
        cache = []
        while self.next_in >= self.next_out:
            _ = self._depart()
            cache.append([self.clock, self.state])
            self.next_out = self._get()

        _ = self._arrive()
        cache.append([self.clock, self.state])
        rslt["inter_arrival"], self.next_in = self.source.__call__()
        rslt["service_time"], rslt["waiting_time"] = self.service_time, self.delay
        rslt["inter_depart"] = self.service_time + self.idle
        
        self._put(self.__proceed(rslt["inter_arrival"]))
        # if np.isinf(self.next_out): self.next_out = self._get()
        self.state_trace = self.state_trace + cache

        if self.clock > self.MAX_T:
            self.reset()
        
        return self.clock, self.state, rslt
    

    def __proceed(self, inter_arrival):
        delta = self.delay + self.service_time - inter_arrival
        self.idle = -min(0, delta)
        self.delay = max(0, delta)
        
        self.service_time, _next_out_ = self.server.__call__(self.idle)
        return _next_out_
    

    def reset(self, *, backT=None):
        if backT is None: backT = self.clock
        _state_trace_cached_ = np.array(self.state_trace)
        _state_trace_cached_[:,0] -= backT
        self.state_trace = _state_trace_cached_.tolist()
        if len(self.schedule) > 0: self.schedule = (np.array(self.schedule) - backT).tolist()
        super().reset(backT=backT)



class ServiceCtrl(QueueBase):

    def __init__(
        self,
        source,
        server,
        *,
        max_T=1e8, 
        begin_busy=False,
        state_load=0,
        work_load=0,
        **kwargs
    ):
        super().__init__(source, server, max_T=max_T, **kwargs)
        # self.arrival_schedule()

        if begin_busy:
            self.state = state_load
            self.delay = work_load
            self.server.clock += work_load
            self.server.workload += work_load
            if state_load > 1:
                self._put(*tuple(np.sort(work_load*np.random.uniform(size=state_load-1))))
            self._put(work_load, work_load+self.next_out)
        self.service_time, _next_out_ = self.server.__call__(self.delay)
        self._put(_next_out_)
        self.next_out = self._get()
        self.state_trace = [[self.clock, self.state]]


    def __call__(self):
        """
        next depart
        """
        rslt = {}
        cache = []

        while self.next_in <= self.next_out:
            _ = self._arrive()
            cache.append([self.clock, self.state])
            inter_arrival, self.next_in = self.source.__call__()
            self._put(inter_arrival)

        _ = self._depart()
        cache.append([self.clock, self.state])
        self.state_trace = self.state_trace + cache
        
        rslt["inter_arrival"] = self._get()
        rslt["service_time"] = self.service_time
        rslt["waiting_time"] = self.delay
        rslt["inter_depart"] = self.idle + self.service_time
        
        self.__proceed(rslt["inter_arrival"])

        if self.clock > self.MAX_T:
            self.reset()
        
        return self.clock, self.state, rslt
    

    def __proceed(self, inter_arrival):
        delta = self.delay + self.service_time - inter_arrival
        self.idle = -min(0, delta)
        self.delay = max(0, delta)
        self.service_time, self.next_out = self.server.__call__(self.idle)


    def reset(self, *, backT=None):
        if backT is None: backT = self.clock
        _state_trace_cached_ = np.array(self.state_trace)
        _state_trace_cached_[:,0] -= backT
        self.state_trace = _state_trace_cached_.tolist()
        super().reset(backT=backT)
