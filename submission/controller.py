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
        """
        Detect obstacles using panoramic vision and adjust turning to avoid dark (close) objects.
        """
        vision = obs["vision"]  # shape: (2, 721, 2)
        # Average across yellow/pale channels → shape: (2, 721)
        brightness_per_eye = np.mean(vision, axis=2)
        
        # Concatenate both eyes → shape: (1442,)
        panoramic_vision = np.concatenate(brightness_per_eye, axis=0)

        # Divide into 5 equal regions
        region_size = panoramic_vision.shape[0] // 5
        brightness = []
        for i in range(5):
            region = panoramic_vision[i * region_size:(i + 1) * region_size]
            brightness.append(np.mean(region))

        # Normalize and invert to get "darkness"
        brightness = np.array(brightness)
        max_brightness = np.max(brightness)
        darkness = 1.0 - brightness / (max_brightness + 1e-6)
        darkness = np.clip(darkness, 0, 1)

        # Compute repulsion: steer away from dark zones
        left = darkness[0] + 0.5 * darkness[1]
        right = darkness[4] + 0.5 * darkness[3]
        repulsion = right - left  # >0 → steer left, <0 → steer right

        avoidance_turn = np.tanh(repulsion * 2.0)  # Gain controls sharpness

        # Create avoidance descending signals
        DELTA_MIN = 0.2
        DELTA_MAX = 1.0
        if avoidance_turn > 0:
            left_signal = DELTA_MAX
            right_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * avoidance_turn
        else:
            right_signal = DELTA_MAX
            left_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * abs(avoidance_turn)

        # Weight the avoidance vs odor commands
        avoidance_importance = float(np.max(darkness))
        odor_importance = 1.0 - avoidance_importance

        combined_left = (
            odor_importance * odor_taxis_command.left_descending_signal
            + avoidance_importance * left_signal
        )
        combined_right = (
            odor_importance * odor_taxis_command.right_descending_signal
            + avoidance_importance * right_signal
        )

        return CommandWithImportance(combined_left, combined_right, 1.0)

    def ball_avoidance(self, obs: Observation) -> CommandWithImportance:
        return CommandWithImportance(0, 0, 0)

    def get_actions(self, obs: Observation) -> Action:
        self.time = self.time + self.timestep
        # Get the odor taxis command
        odor_taxis_command = self.get_odor_taxis(obs)
        #get the pillar avoidance command
        pillar_avoidance_command = self.pillar_avoidance(obs, odor_taxis_command)


        action=np.array([
            odor_taxis_command.left_descending_signal,
            odor_taxis_command.right_descending_signal,
            pillar_avoidance_command.left_descending_signal,
            pillar_avoidance_command.right_descending_signal,
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
