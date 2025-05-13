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

        self.position = np.array([0.0, 0.0])
        self.position_trace = []
        self.drift_trace = []  # log drift over time

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.timestep = timestep
        self.time = 0

        # Data storage
        self.joint_angles_over_time = []
        self.leg_positions_over_time = []

        # Path integration state
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]
        self.leg_displacement_used = [0] * 6  # <-- keep track of used displacements

        # Heading estimation A CHANGER 
        self.heading = 0.0
        self.heading_gain = 0.04
        self.max_heading_change = np.deg2rad(5)
        self.last_delta_heading = 0.0
        self.smooth_alpha = 0.6 # exponential smoothing factor

        # Logging
        self.heading_trace = []

    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
        # unchanged odor taxis logic
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

    def estimate_heading_change(self):
        # Fixed tripod: legs [0,3,4] vs [1,2,5]
        # Ce code fonctionne donc bien sur le tripod gait 
        left_indices = [0, 3, 4]
        right_indices = [1, 2, 5]
        left_disp, right_disp = [], []

        for i in left_indices:
            if self.leg_displacements[i]:
                d = self.leg_displacements[i][-1][0]
                if abs(d) > 1e-4:
                    left_disp.append(d)
        for i in right_indices:
            if self.leg_displacements[i]:
                d = self.leg_displacements[i][-1][0]
                if abs(d) > 1e-4:
                    right_disp.append(d)

        if left_disp and right_disp:
            raw = (np.mean(right_disp) - np.mean(left_disp)) * self.heading_gain
            raw = np.clip(raw, -self.max_heading_change, self.max_heading_change)
            # exponential smoothing
            delta = self.smooth_alpha * raw + (1 - self.smooth_alpha) * self.last_delta_heading
            self.last_delta_heading = delta
            return delta
        return 0.0

    def update_position(self, stance_phases):
        # Ne pas réutiliser des déplacements déjà comptés
        total_disp = np.zeros(2)
        count = 0
        for i in range(6):
            used_idx = self.leg_displacement_used[i]
            disps = self.leg_displacements[i]
            while used_idx < len(disps):
                disp = disps[used_idx][:2]
                if np.linalg.norm(disp) > 1e-4:
                    total_disp += disp
                    count += 1
                self.leg_displacement_used[i] += 1
                used_idx += 1
        if count == 0:
            return
        avg_disp = total_disp / count
        # Foot displacement is opposite to body movement: invert
        rot = np.array([
            [np.cos(self.heading), -np.sin(self.heading)],
            [np.sin(self.heading),  np.cos(self.heading)]
        ])
        dw = -rot.dot(avg_disp)
        self.position += dw
        self.position_trace.append(self.position.copy())
        print(f"Estimated position: {self.position} (delta_world={dw})")

    def get_actions(self, obs: Observation) -> Action:
        self.time += self.timestep
        odor = self.get_odor_taxis(obs)
        action = np.array([odor.left_descending_signal, odor.right_descending_signal])
        joint_angles, adhesion = step_cpg(self.cpg_network, self.preprogrammed_steps, action)
        self.joint_angles_over_time.append(joint_angles)
        legs = obs["end_effectors"]
        forces = obs["contact_forces"]
        stance = [np.linalg.norm(f) > 4 for f in forces]

        for i in range(6):
            if stance[i] and not self.leg_stance_history[i]:
                self.leg_stance_start_positions[i] = np.array(legs[i])
            if not stance[i] and self.leg_stance_history[i]:
                start = self.leg_stance_start_positions[i]
                if start is not None:
                    disp = np.array(legs[i]) - start
                    disp = np.clip(disp, -5, 5)  # Clip displacement to prevent outliers
                    self.leg_displacements[i].append(disp)
                    print(f"Leg {i} displacement during stance: {disp}")
                self.leg_stance_start_positions[i] = None
            self.leg_stance_history[i] = stance[i]

        # Vient de l'approximation des membres. Pas utile pour debug

        dh = self.estimate_heading_change()
        self.heading = (self.heading + dh + np.pi) % (2 * np.pi) - np.pi

        #Utiliser pour verifier les beug
        # self.heading = obs["heading"]
        # self.position = np.array(obs["fly"][0][:2])

        self.update_position(stance)

        # print("\n--- Debug Info ---")
        # print(f"Time: {self.time:.4f} | Legs in stance: {sum(stance)}/6")
        # print(f"Stance flags: {stance}")
        # print(f"Estimated heading: {np.rad2deg(self.heading):.2f}°")
        if "fly" in obs:
            print(f"True position: {obs['fly'][0][:2]}, True heading: {np.rad2deg(obs['heading']):.2f}°")
            # log drift
            true_pos = np.array(obs["fly"][0][:2])
            est_pos = self.position
            drift = np.linalg.norm(true_pos - est_pos)
            self.drift_trace.append(drift)

        # log heading
        self.heading_trace.append(self.heading)
        return {"joints": joint_angles, "adhesion": adhesion}

    def done_level(self, obs: Observation):
        return False

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.joint_angles_over_time = []
        self.leg_positions_over_time = []
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]
        self.leg_displacement_used = [0] * 6
        self.heading = 0.0
        self.position = np.array([0.0, 0.0])
        self.position_trace = []
        self.drift_trace = []
        self.last_delta_heading = 0.0

    def print_total_leg_displacements(self):
        print("\n--- Total leg displacements ---")
        for i, dlist in enumerate(self.leg_displacements):
            tot = np.sum(dlist, axis=0) if dlist else np.array([0, 0])
            print(f"Leg {i}: Total displacement vector = {tot}")
        print(f"Final estimated position: {self.position}")
