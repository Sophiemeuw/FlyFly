import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple

class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float # 0->1


class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.timestep = timestep
        self.time = 0
    
    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
        ODOR_GAIN = -500 # need to be tuned
        # min and max range of the descending signal
        DELTA_MIN = 0.2 
        DELTA_MAX = 1
     
        I_right = ((obs["odor_intensity"][0][1] + obs["odor_intensity"][0][3]))/2
        I_left = ((obs["odor_intensity"][0][0] + obs["odor_intensity"][0][2]))/2
        assymmetry = (I_left - I_right) /((I_left + I_right + 1e-6)/2)
        s = assymmetry * ODOR_GAIN
        # print(s)
        turning_bias = np.tanh(s**2)

        if s > 0: 
            right_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            left_descending_signal = DELTA_MAX
        else: 
            left_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            right_descending_signal = DELTA_MAX

        return CommandWithImportance(left_descending_signal, right_descending_signal, 0.5)
    
    def pillar_avoidance(self, obs: Observation, odor_taxis_command: CommandWithImportance) -> CommandWithImportance:
        return odor_taxis_command

    def ball_avoidance(self, obs: Observation) -> CommandWithImportance:
        return CommandWithImportance(0, 0, 0)

    def get_actions(self, obs: Observation) -> Action:
        self.time = self.time + self.timestep
        # Get the odor taxis command
        odor_taxis_command = self.get_odor_taxis(obs)

        action=np.array([
            odor_taxis_command.left_descending_signal,
            odor_taxis_command.right_descending_signal,
        ])

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
