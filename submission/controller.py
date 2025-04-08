import numpy as np
from cobar_miniproject.base_controller import BaseController
import random
from .utils import get_cpg, step_cpg




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

    def get_actions(self, obs):
        """
        Determine the fly's actions based on odor taxis behavior.
        """
        # Check if the fly senses an odor gradient
        odor_intensity = obs["odor_intensity"]
        attractive_intensity = np.sum(odor_intensity[0, :])
        aversive_intensity = np.sum(odor_intensity[1, :])

        # If no significant gradient is detected, turn in place
        if attractive_intensity < 1e-2 and aversive_intensity < 1e-2:
            control_signal = np.array([1.0, -1.0]) 
        else:
            # Follow the odor gradient using odor taxis logic
            control_signal = self.get_odor_taxis_actions(obs, odor_intensity)

        print(f"attractive:{attractive_intensity} aversive:{aversive_intensity} control signal{control_signal}")
        # Use the control signal to compute joint angles and adhesion
        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=control_signal,
        )

        # Return the required action dictionary
        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }
    
    def get_odor_taxis_actions(self, obs, odor_gradient):
        """
        Compute actions based on odor taxis behavior.
        """
        attractive_gain = -500
        aversive_gain = 80
        delta_min = 0.2
        delta_max = 1.0

        # Compute left-right asymmetry for attractive and aversive odors
        attractive_intensities = np.average(
            obs["odor_intensity"][0, :].reshape(2, 2), axis=0, weights=[9, 1]
        )
        aversive_intensities = np.average(
            obs["odor_intensity"][1, :].reshape(2, 2), axis=0, weights=[10, 0]
        )
        attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / np.maximum(attractive_intensities.mean(), 1e-6)
        )
        aversive_bias = (
            aversive_gain
            * (aversive_intensities[0] - aversive_intensities[1])
            / np.maximum(aversive_intensities.mean(), 1e-6)
        )
        effective_bias = attractive_bias + aversive_bias

        # Apply nonlinear transformation to the effective bias
        effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)

        # Modulate the control signal based on the transformed bias
        control_signal = np.ones((2,))
        if effective_bias_norm > 0:
            control_signal[0] = delta_max - effective_bias_norm * (delta_max - delta_min)
            control_signal[1] = delta_max
        else:
            control_signal[0] = delta_max
            control_signal[1] = delta_max - abs(effective_bias_norm) * (delta_max - delta_min)

        return control_signal
    
    def done_level(self, obs):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
