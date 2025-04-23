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
        IMPORTANCE = 0.6 # Odor taxis has lower priority
     
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

        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)
    
    def pillar_avoidance(self, obs: Observation, odor_taxis_command: CommandWithImportance) -> CommandWithImportance:
        #if not obs.get("vision_updated", False):
            #return odor_taxis_command  # skip if vision wasn't updated
        #Parameters :
        MAX_DELTA = 1.0
        MIN_DELTA = 0.2
        IMPORTANCE = 0.6  # Avoidance has higher priority

        vision = obs["vision"]  # shape: (2, 721, 2)
        brightness = np.mean(vision, axis=2)  # shape: (2, 721)

        # Weight ommatidia: front gets higher weight
        weights = np.linspace(1.0, 0.5, 721)

        left_weighted = np.sum(brightness[0] * weights)
        right_weighted = np.sum(brightness[1] * weights)

        # Compute turning signal based on difference
        diff = left_weighted - right_weighted
        gain = 3.0
        turn_signal = np.tanh(gain * diff)

        # Adjust descending signals
        
        left_signal = MAX_DELTA
        right_signal = MAX_DELTA

        if turn_signal > 0:
            # Obstacle on the left → turn right
            left_signal = MAX_DELTA - (MAX_DELTA - MIN_DELTA) * abs(turn_signal)
        else:
            # Obstacle on the right → turn left
            right_signal = MAX_DELTA - (MAX_DELTA - MIN_DELTA) * abs(turn_signal)

        # Combine with odor taxis command via IMPORTANCE weighting
        

        left_descending_signal = (
            IMPORTANCE * left_signal + (1 - IMPORTANCE) * odor_taxis_command.left_descending_signal
        )
        right_descending_command = (
            IMPORTANCE * right_signal + (1 - IMPORTANCE) * odor_taxis_command.right_descending_signal
        )

        return CommandWithImportance(left_descending_signal, right_descending_command, IMPORTANCE)


    def ball_avoidance(self, obs: Observation) -> CommandWithImportance:
        return CommandWithImportance(0, 0, 0)

    def get_actions(self, obs: Observation) -> Action:
        self.time = self.time + self.timestep
        # Get the odor taxis command
        odor_taxis_command = self.get_odor_taxis(obs)
        # Get combined pillar avoidance + odor taxis command
        combined_command = self.pillar_avoidance(obs, odor_taxis_command)

        # Only use the 2 blended signals
        action = np.array([
            combined_command.left_descending_signal,
            combined_command.right_descending_signal,
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
