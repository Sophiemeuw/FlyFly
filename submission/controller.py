import numpy as np
import random
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
        turning_bias = np.abs(np.tanh(s))


        if s > 0:
            # Turn left (more odor on left)
            right_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            left_descending_signal = DELTA_MAX
        else:
            # Turn right (more odor on right)
            left_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            right_descending_signal = DELTA_MAX

        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)
    
    def pillar_avoidance(self, obs: Observation, odor_taxis_command: CommandWithImportance) -> CommandWithImportance:
        MAX_DELTA = 1.0
        MIN_DELTA = 0.2
        IMPORTANCE = 0.7
        GAIN = 15500  # Increased sensitivity

        vision = obs["vision"]  # shape: (2, 721, 2)
        brightness = np.mean(vision, axis=2)  # shape: (2, 721)

        # Weight ommatidia: front gets higher weight
        center = 360
        std = 120
        #weights = np.exp(-((np.arange(721) - center) ** 2) / (2 * std**2))

        left_weighted = np.sum(brightness[0])
        right_weighted = np.sum(brightness[1])

        # Compute binocular front brightness (center ~721)
        left_front = brightness[0, 620:]
        right_front = brightness[1, :100]
        front_overlap_brightness = np.mean(np.concatenate([left_front, right_front]))

        # Detect side occlusion (darkness to the left or right)
        left_side_brightness = np.mean(brightness[0, 500:620])
        right_side_brightness = np.mean(brightness[1, 100:220])

        # Trigger emergency swerve if stuck and sees darkness in front or on the side
        if (
            front_overlap_brightness < 10
            or left_side_brightness < 3
            or right_side_brightness < 3
        ) and np.linalg.norm(obs["velocity"][:2]) < 0.2:

            # Swerve away from the darker side
            if left_side_brightness < right_side_brightness:
                left_signal = 0.1
                right_signal = 1.0
            elif right_side_brightness < left_side_brightness:
                left_signal = 1.0
                right_signal = 0.1
            else:
                # Fall back to random left or right if sides are similar
                if random.random() < 0.5:
                    left_signal = 1.0
                    right_signal = 0.1
                else:
                    left_signal = 0.1
                    right_signal = 1.0

            return CommandWithImportance(
                IMPORTANCE * left_signal + (1 - IMPORTANCE) * odor_taxis_command.left_descending_signal,
                IMPORTANCE * right_signal + (1 - IMPORTANCE) * odor_taxis_command.right_descending_signal,
                IMPORTANCE
            )

        # Compute turning signal based on weighted side brightness
        diff = left_weighted - right_weighted
        turn_signal = np.tanh(GAIN * diff)

        # Add mild jitter to avoid getting stuck in symmetric situations
        turn_signal += np.random.uniform(-0.05, 0.05)

        # Convert turn signal to descending motor commands
        left_signal = MAX_DELTA
        right_signal = MAX_DELTA
        if turn_signal > 0:
            left_signal = MAX_DELTA - (MAX_DELTA - MIN_DELTA) * abs(turn_signal)
        else:
            right_signal = MAX_DELTA - (MAX_DELTA - MIN_DELTA) * abs(turn_signal)

        # Combine with odor taxis command
        left_descending_signal = (
            IMPORTANCE * left_signal + odor_taxis_command.importance * odor_taxis_command.left_descending_signal
        )
        right_descending_signal = (
            IMPORTANCE * right_signal + odor_taxis_command.importance * odor_taxis_command.right_descending_signal
        )
       
        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)



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