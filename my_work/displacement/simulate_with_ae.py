from pyexadis_base import SimulateNetwork


class SimulateNetworkWithAE(SimulateNetwork):
    """
    在基类 SimulateNetwork 基础上，额外接入 AEDisplacementRecorder，
    在 step_begin 存旧位置，在 step_post_integrate 算并记录位移增量。
    """
    def __init__(self, *args, ae_recorder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ae_recorder = ae_recorder

    def step_begin(self, N, state):
        super().step_begin(N, state)
        if self.ae_recorder is not None:
            self.ae_recorder.snapshot_before_step(N.get_disnet())

    def step_post_integrate(self, N, state):
        super().step_post_integrate(N, state)
        if self.ae_recorder is not None:
            self.ae_recorder.record_step(N.get_disnet())