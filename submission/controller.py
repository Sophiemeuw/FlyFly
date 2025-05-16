import numpy as np
import random
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple


class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float  # 0->1


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.timestep = timestep
        self.time = 0.0

        # Odor taxis setup
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()

        # COM state (meters)
        self.position = np.zeros(2)
        # empirical gain for displacement integration (0 < g < 1)
        self.g = 0.3

        # Path integration state
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]

        # Logging
        self.position_trace = []

    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
        ODOR_GAIN, DELTA_MIN, DELTA_MAX, IMPORTANCE = -500, 0.2, 1.0, 0.5
        I = obs["odor_intensity"][0]
        I_left = (I[0] + I[2]) / 2
        I_right = (I[1] + I[3]) / 2
        asym = (I_left - I_right) / ((I_left + I_right + 1e-6) / 2)
        s = asym * ODOR_GAIN
        bias = abs(np.tanh(s))
        if s > 0:
            right = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * bias
            left = DELTA_MAX
        else:
            left = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * bias
            right = DELTA_MAX
        return CommandWithImportance(left, right, IMPORTANCE)

    def integrate_displacement(self, obs: Observation):
        # Collect new stance-phase displacements
        legs = obs["end_effectors"]
        forces = obs["contact_forces"]
        stance = [np.linalg.norm(f) > 4 for f in forces]

        for i in range(6):
            if stance[i] and not self.leg_stance_history[i]:
                self.leg_stance_start_positions[i] = np.array(legs[i])
            if not stance[i] and self.leg_stance_history[i]:
                start = self.leg_stance_start_positions[i]
                if start is not None:
                    disp3 = np.array(legs[i]) - start
                    disp2 = np.clip(disp3[:2], -5, 5)
                    self.leg_displacements[i].append(disp2)
                self.leg_stance_start_positions[i] = None
            self.leg_stance_history[i] = stance[i]

        # Sum and average displacements
        total_disp = np.zeros(2)
        count = 0
        for dlist in self.leg_displacements:
            for disp in dlist:
                total_disp += disp
                count += 1
        if count == 0:
            return
        avg_disp = total_disp / count

        # Rotate displacement by heading if available
        if "heading" in obs:
            # assume obs['heading'][0] gives yaw angle in radians
            yaw = obs['heading']  # heading is provided as a scalar yaw angle in radians
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            avg_disp = R.dot(avg_disp)

        # Integrate with empirical gain
        dw = -self.g * avg_disp
        self.position += dw

        # Clear used displacements
        self.leg_displacements = [[] for _ in range(6)]

    def get_actions(self, obs: Observation) -> Action:
        self.time += self.timestep
        cmd = self.get_odor_taxis(obs)
        action = np.array([cmd.left_descending_signal, cmd.right_descending_signal])

        joints, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action,
        )

        # Integrate COM via empirical displacement gain
        self.integrate_displacement(obs)

        # Log position (meters)
        self.position_trace.append(self.position.copy())

        # Debug print (SI units, no mm conversion, includes yaw)
        if "fly" in obs:
            true_xy = obs['fly'][0][:2]
            print(f"Time: {self.time:.4f} s | True pos: {true_xy} m | Est pos: {self.position} m")

        return {"joints": joints, "adhesion": adhesion}

    def done_level(self, obs: Observation):
        return False

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.position[:] = 0
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]
        self.position_trace = []

    def print_trajectory(self):
        print("Position trace (m):", self.position_trace)
